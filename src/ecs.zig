const std = @import("std");
const bit_tree = @import("bit_tree");

const Tree = bit_tree.BitTree(.u64);
const BitState = bit_tree.BitState;
const Allocator = std.mem.Allocator;

/// Fixed-identity table tag is the `tag` comptime parameter itself:
/// every `EntityReference` belongs to its `ECSTable(tag)` instantiation by type.
/// `tag` must be an enum literal (e.g. `.default`, `.main`), so different tags
/// give fully isolated table instances for the same component set.
pub fn ECSTable(comptime tag: @EnumLiteral()) type {
    return struct {
        const Table = @This();

        var initialized: bool = false;
        var destroyed: Tree = .{};
        var activity: Tree = .{};
        var cleared: Tree = .{};
        var generations: std.ArrayListUnmanaged(u8) = .empty;
        var indirection: std.ArrayListUnmanaged(u32) = .empty;
        var row_to_slot: std.ArrayListUnmanaged(u32) = .empty;
        var data_cols: std.ArrayListUnmanaged(DataColumn) = .empty;
        var flag_cols: std.ArrayListUnmanaged(FlagColumn) = .empty;
        var data_map: std.StringHashMapUnmanaged(u32) = .empty;
        var flag_map: std.StringHashMapUnmanaged(u32) = .empty;

        /// Table identity carried by the type, never stored per entity.
        pub const tag_value = tag;

        /// Component column with payload. Bytes are typeless `u8` storage
        /// with an explicit cast on request; layout is base-aligned with a
        /// size-rounded stride so every row address satisfies `@alignOf(T)`.
        const DataColumn = struct {
            /// Owned `@typeName(T)` copy, map key for the column.
            name: []u8,
            /// Payload size in bytes, `@sizeOf(T)`.
            size: usize,
            /// Row stride in bytes, size rounded up to alignment.
            stride: usize,
            /// Base alignment in bytes, `@alignOf(T)`, always a power of two.
            alignment: usize,
            /// Per-row activity flags.
            tree: Tree = .{},
            /// Owned allocation backing `buf`, freed as a whole.
            raw: []u8 = &.{},
            /// Aligned window into `raw`: packed rows, base-aligned.
            buf: []u8 = &.{},
            /// Allocated row capacity.
            cap: usize = 0,
            /// Live row count, mirrors the entity planes.
            len: usize = 0,
        };

        /// Flag-only component column. No payload, just activity flags.
        /// `dummy` provides a stable address for `getComponentPtr` on flag
        /// types: empty structs carry no bytes, so every row of one flag
        /// shares this single byte. Writes through the pointer are no-ops;
        /// activity is tracked only in `tree`. Never compare pointers for
        /// row identity and never store them across `removeComponent`.
        const FlagColumn = struct {
            /// Owned `@typeName(T)` copy, map key for the column.
            name: []u8,
            /// Per-row activity flags.
            tree: Tree = .{},
            /// Shared addressable byte for flag `getComponentPtr`.
            dummy: u8 = 0,
        };

        /// Failure modes of table operations, including allocator failures.
        pub const Error = Allocator.Error || error{
            /// Table was not initialized with `init`.
            NotInitialized,
            /// Table is already initialized, `deinit` first.
            AlreadyInitialized,
            /// Slot is out of range, generation mismatched or slot cleared.
            InvalidReference,
            /// Row is out of range.
            RowOutOfBounds,
            /// Entity or row is already marked destroyed.
            AlreadyDestroyed,
            /// Component type is not registered in the table.
            ComponentNotFound,
            /// Component type is already registered in the table.
            ComponentAlreadyExists,
        };

        /// Stable entity handle. Holds a slot index into the grow-only
        /// indirection table, never a direct row: rows move on
        /// `clearDestroyed` while slots stay valid for the table lifetime.
        pub const EntityReference = struct {
            /// Index into the indirection table.
            slot: u32,
            /// Slot generation, bumped on every `destroy`.
            gen: u8,

            /// Checks the handle against slot bounds, generation, cleared
            /// flag and the destroyed flag of the target row.
            pub fn isValid(self: EntityReference) bool {
                return Table.isValidRef(self);
            }

            /// Marks the entity destroyed and bumps the slot generation.
            /// The row is reclaimed later by `clearDestroyed`.
            pub fn destroy(self: EntityReference) Error!void {
                const row = try Table.resolveRef(self);
                Table.destroyRow(row);
            }

            /// Reads the entity activity flag.
            pub fn isActive(self: EntityReference) Error!bool {
                const row = try Table.resolveRef(self);
                return Row.isActive(row);
            }

            /// Writes the entity activity flag.
            pub fn setActivity(self: EntityReference, state: BitState) Error!void {
                const row = try Table.resolveRef(self);
                return Row.setActivity(row, state);
            }

            /// Copies the component payload out of the entity row.
            /// Flag components return `T{}`.
            pub fn getComponent(self: EntityReference, comptime T: type) Error!T {
                return Row.getComponent(try Table.resolveRef(self), T);
            }

            /// Returns a mutable pointer into the entity row payload.
            /// Flag components share one address per column, writes are no-ops.
            pub fn getComponentPtr(self: EntityReference, comptime T: type) Error!*T {
                return Row.getComponentPtr(try Table.resolveRef(self), T);
            }

            /// Overwrites the component payload of the entity row.
            /// The value type selects the column, no separate type tag.
            /// Flag values only set the activity bit.
            pub fn setComponent(self: EntityReference, component: anytype) Error!void {
                return Row.setComponent(try Table.resolveRef(self), component);
            }

            /// Reads the component activity flag of the entity.
            pub fn isComponentActive(self: EntityReference, comptime T: type) Error!bool {
                return Row.isComponentActive(try Table.resolveRef(self), T);
            }

            /// Writes the component activity flag of the entity.
            pub fn setComponentActivity(self: EntityReference, comptime T: type, state: BitState) Error!void {
                return Row.setComponentActivity(try Table.resolveRef(self), T, state);
            }

            /// Reads the activity flag of a registered `@EnumLiteral()` flag.
            pub fn isFlagActive(self: EntityReference, comptime flag: @EnumLiteral()) Error!bool {
                return Row.isFlagActive(try Table.resolveRef(self), flag);
            }

            /// Writes the activity flag of a registered `@EnumLiteral()` flag.
            pub fn setFlagActivity(self: EntityReference, comptime flag: @EnumLiteral(), state: BitState) Error!void {
                return Row.setFlagActivity(try Table.resolveRef(self), flag, state);
            }

            /// Enables a registered `@EnumLiteral()` flag on the entity.
            pub fn setFlag(self: EntityReference, comptime flag: @EnumLiteral()) Error!void {
                return Row.setFlag(try Table.resolveRef(self), flag);
            }
        };

        /// Raw row operations without indirection. Rows are unstable across
        /// `clearDestroyed` and reuse: use only inside component iteration,
        /// never store row ids. Every method validates bounds explicitly.
        pub const Row = struct {
            /// Checks that the row exists and is not marked destroyed.
            pub fn isAlive(row: u32) bool {
                if (!initialized) return false;
                if (row >= destroyed.bitset.bits_count) return false;
                return destroyed.bitset.getBit(row) == .inactive;
            }

            /// Marks a live row destroyed and bumps its slot generation.
            pub fn destroy(row: u32) Error!void {
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                Table.destroyRow(row);
            }

            /// Reads the entity activity flag of a live row.
            pub fn isActive(row: u32) Error!bool {
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                return activity.bitset.getBit(row) == .active;
            }

            /// Writes the entity activity flag of a live row.
            pub fn setActivity(row: u32, state: BitState) Error!void {
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                activity.setBit(row, state);
            }

            /// Copies the component payload out of a live row.
            /// Returned bytes are meaningful only while the row is active.
            /// Flag components carry no payload and return `T{}`; use
            /// `isComponentActive` or `Query` to distinguish enabled rows.
            pub fn getComponent(row: u32, comptime T: type) Error!T {
                if (comptime !hasFields(T)) {
                    if (!initialized) return Error.NotInitialized;
                    if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                    if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                    _ = try Table.flagColumn(T);
                    return T{};
                }
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                const col = try Table.dataColumn(T);
                var out: T = undefined;
                @memcpy(std.mem.asBytes(&out), Table.rowBytes(col, row));
                return out;
            }

            /// Returns a mutable pointer into a live row payload.
            /// The row address satisfies `@alignOf(T)` by column layout.
            /// Flag components share one `dummy` byte per column: every row
            /// of one flag returns the same address, writes are no-ops.
            /// Never compare these pointers for row identity.
            pub fn getComponentPtr(row: u32, comptime T: type) Error!*T {
                if (comptime !hasFields(T)) {
                    if (!initialized) return Error.NotInitialized;
                    if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                    if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                    const col = try Table.flagColumn(T);
                    return @ptrCast(@alignCast(&col.dummy));
                }
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                const col = try Table.dataColumn(T);
                return @ptrCast(@alignCast(Table.rowBytes(col, row).ptr));
            }

            /// Overwrites the component payload of a live row.
            /// The value type selects the column, no separate type tag.
            /// Flag values (e.g. `Tag{}`) and `@EnumLiteral()` flags
            /// (e.g. `.my_flag`) only set the activity bit.
            pub fn setComponent(row: u32, component: anytype) Error!void {
                if (comptime isEnumType(@TypeOf(component))) {
                    if (!initialized) return Error.NotInitialized;
                    if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                    if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                    const tree = try Table.enumFlagTree(component);
                    tree.setBit(row, .active);
                    return;
                }
                const T = @TypeOf(component);
                if (comptime !hasFields(T)) {
                    if (!initialized) return Error.NotInitialized;
                    if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                    if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                    const tree = try Table.componentTree(T);
                    tree.setBit(row, .active);
                    return;
                }
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                const col = try Table.dataColumn(T);
                @memcpy(Table.rowBytes(col, row), std.mem.asBytes(&component));
            }

            /// Reads the component activity flag of a live row.
            /// Works for payload and flag components alike.
            pub fn isComponentActive(row: u32, comptime T: type) Error!bool {
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                const tree = try Table.componentTree(T);
                return tree.bitset.getBit(row) == .active;
            }

            /// Writes the component activity flag of a live row.
            /// Works for payload and flag components alike.
            pub fn setComponentActivity(row: u32, comptime T: type, state: BitState) Error!void {
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                const tree = try Table.componentTree(T);
                tree.setBit(row, state);
            }

            /// Reads the activity flag of a registered `@EnumLiteral()` flag.
            /// Parallel API to `isComponentActive`, which stays `type`-only
            /// to preserve ZLS hints: `isComponentActive(Tag)` vs
            /// `isFlagActive(.my_flag)`.
            pub fn isFlagActive(row: u32, comptime flag: @EnumLiteral()) Error!bool {
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                const tree = try Table.enumFlagTree(flag);
                return tree.bitset.getBit(row) == .active;
            }

            /// Writes the activity flag of a registered `@EnumLiteral()` flag.
            /// Parallel API to `setComponentActivity`.
            pub fn setFlagActivity(row: u32, comptime flag: @EnumLiteral(), state: BitState) Error!void {
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                const tree = try Table.enumFlagTree(flag);
                tree.setBit(row, state);
            }

            /// Enables a registered `@EnumLiteral()` flag on a live row.
            /// Sugar over `setFlagActivity(row, flag, .active)`; the
            /// `setComponent(.flag)` value path does the same during `create`.
            pub fn setFlag(row: u32, comptime flag: @EnumLiteral()) Error!void {
                try setFlagActivity(row, flag, .active);
            }
        };

        /// Resolves the activity tree of a registered component,
        /// choosing the payload or flag column at comptime.
        fn componentTree(comptime T: type) Error!*Tree {
            if (hasFields(T)) {
                const col = try dataColumn(T);
                return &col.tree;
            } else {
                const col = try flagColumn(T);
                return &col.tree;
            }
        }

        /// Resolves a flag column by component type.
        fn flagColumn(comptime T: type) Error!*FlagColumn {
            const idx = flag_map.get(@typeName(T)) orelse return Error.ComponentNotFound;
            return &flag_cols.items[idx];
        }

        /// Resolves a payload column by component type.
        fn dataColumn(comptime T: type) Error!*DataColumn {
            const idx = data_map.get(@typeName(T)) orelse return Error.ComponentNotFound;
            return &data_cols.items[idx];
        }

        /// Addresses the payload bytes of one row: `size` bytes inside
        /// the row stride, base-aligned by column layout.
        fn rowBytes(col: *DataColumn, row: u32) []u8 {
            const base: usize = @as(usize, row) * col.stride;
            return col.buf[base .. base + col.size];
        }

        /// Marks the table usable. Must precede any other call.
        /// Takes no allocator: the empty table owns nothing yet.
        pub fn init() Error!void {
            if (initialized) return Error.AlreadyInitialized;
            initialized = true;
        }

        /// Releases every plane and index. All handles die with the table:
        /// indirection and cleared storage are freed, never shrunk before.
        pub fn deinit(alloc: Allocator) void {
            if (!initialized) return;
            destroyed.deinit(alloc);
            destroyed = .{};
            activity.deinit(alloc);
            activity = .{};
            cleared.deinit(alloc);
            cleared = .{};
            generations.deinit(alloc);
            generations = .empty;
            indirection.deinit(alloc);
            indirection = .empty;
            row_to_slot.deinit(alloc);
            row_to_slot = .empty;
            for (data_cols.items) |*col| freeDataColumn(alloc, col);
            data_cols.deinit(alloc);
            data_cols = .empty;
            for (flag_cols.items) |*col| {
                col.tree.deinit(alloc);
                alloc.free(col.name);
            }
            flag_cols.deinit(alloc);
            flag_cols = .empty;
            data_map.deinit(alloc);
            data_map = .empty;
            flag_map.deinit(alloc);
            flag_map = .empty;
            initialized = false;
        }

        /// Compile-time split between payload and flag components.
        /// Non-struct types are rejected; empty structs become flags.
        fn hasFields(comptime T: type) bool {
            const ti = @typeInfo(T);
            if (ti != .@"struct") @compileError("component must be a struct type");
            return ti.@"struct".field_names.len > 0;
        }

        /// True for tuple values/types (structs with `is_tuple`), including
        /// empty `. {}`. Never errors: non-structs return false.
        /// Checked before `hasFields` so empty groups are not mistaken
        /// for flag components.
        fn isTuple(comptime T: type) bool {
            const ti = @typeInfo(T);
            return ti == .@"struct" and ti.@"struct".is_tuple;
        }

        /// True for `@EnumLiteral()` flag values (e.g. `.my_flag`).
        /// Takes a value (not a type): struct component values return
        /// false, struct types return false, only enum literals return true.
        /// Only call in comptime-known contexts (query tuples, flag params):
        /// for runtime component values use `isEnumType(@TypeOf(v))`.
        fn isEnumValue(v: anytype) bool {
            return @typeInfo(@TypeOf(v)) == .enum_literal;
        }

        /// True for the `@EnumLiteral()` type itself. Takes a type, so it
        /// is safe for runtime component values: `isEnumType(@TypeOf(v))`
        /// needs only the (comptime) type, never the (runtime) value.
        fn isEnumType(comptime T: type) bool {
            return @typeInfo(T) == .enum_literal;
        }

        /// Storage key for an `@EnumLiteral()` flag.
        /// All enum literals share one type, so identity is the tag name.
        /// The `@EnumLiteral(."...")` prefix can never collide with
        /// `@typeName(T)` of struct components.
        fn enumFlagKey(comptime flag: @EnumLiteral()) []const u8 {
            return "@EnumLiteral(." ++ @tagName(flag) ++ ")";
        }

        /// Resolves an `@EnumLiteral()` flag column by tag name.
        fn enumFlagColumn(comptime flag: @EnumLiteral()) Error!*FlagColumn {
            const idx = flag_map.get(enumFlagKey(flag)) orelse return Error.ComponentNotFound;
            return &flag_cols.items[idx];
        }

        /// Resolves the activity tree of a registered `@EnumLiteral()` flag.
        fn enumFlagTree(comptime flag: @EnumLiteral()) Error!*Tree {
            const col = try enumFlagColumn(flag);
            return &col.tree;
        }

        /// Number of leaf components in a (possibly nested) query tuple,
        /// depth-first, empty groups contribute zero. Leaves may be struct
        /// component types or `@EnumLiteral()` flags. Duplicates count
        /// repeatedly here; see `countUniqueQueryLeaves`.
        pub fn countQueryLeaves(comptime nested: anytype) usize {
            comptime var n: usize = 0;
            inline for (nested) |e| {
                if (comptime @TypeOf(e) == type) {
                    n += 1;
                } else if (comptime isEnumValue(e)) {
                    n += 1;
                } else if (comptime isTuple(@TypeOf(e))) {
                    n += comptime countQueryLeaves(e);
                } else {
                    @compileError("query tuples hold component types, @EnumLiteral() flags, or nested tuples thereof");
                }
            }
            return n;
        }

        /// idx-th leaf type of a nested query tuple, depth-first. This is
        /// the flat view: no intermediate tuple is ever materialized.
        /// Kept for back-compat with struct-only queries; mixed
        /// struct/enum queries resolve via `queryLeafKeyAt` /
        /// `queryLeafTreeAt` below, which have concrete return types
        /// (`[]const u8` / `Error!*Tree`) so ZLS hints for `T: type`
        /// are preserved and Zig 0.16 accepts them.
        pub fn queryLeafAt(comptime nested: anytype, comptime idx: usize) type {
            comptime var cur: usize = 0;
            inline for (nested) |e| {
                if (comptime @TypeOf(e) == type) {
                    if (cur == idx) return e;
                    cur += 1;
                } else if (comptime isTuple(@TypeOf(e))) {
                    const m = comptime countQueryLeaves(e);
                    // Nested groups may hold enum flags; count them for the
                    // offset but resolve struct leaves only. Mixed callers
                    // must use `queryLeafKeyAt` / `queryLeafTreeAt`.
                    if (idx < cur + m) return comptime queryLeafAt(e, idx - cur);
                    cur += m;
                } else {
                    @compileError("queryLeafAt supports struct component types only; use queryLeafKeyAt/queryLeafTreeAt for mixed struct/@EnumLiteral() tuples");
                }
            }
            @compileError("leaf index out of bounds");
        }

        /// idx-th leaf storage key of a nested query tuple, depth-first:
        /// `@typeName(T)` for struct component types,
        /// `"@EnumLiteral(.name)"` for enum flags. Keys uniquely identify
        /// leaves (all enum literals share one type, so tag name is the
        /// identity), so string equality replaces type `==` for mixed
        /// tuples. Concrete `[]const u8` return keeps Zig 0.16 happy.
        pub fn queryLeafKeyAt(comptime nested: anytype, comptime idx: usize) []const u8 {
            comptime var cur: usize = 0;
            inline for (nested) |e| {
                if (comptime @TypeOf(e) == type) {
                    if (cur == idx) return @typeName(e);
                    cur += 1;
                } else if (comptime isEnumValue(e)) {
                    if (cur == idx) return comptime enumFlagKey(e);
                    cur += 1;
                } else if (comptime isTuple(@TypeOf(e))) {
                    const m = comptime countQueryLeaves(e);
                    if (idx < cur + m) return comptime queryLeafKeyAt(e, idx - cur);
                    cur += m;
                } else {
                    @compileError("query tuples hold component types, @EnumLiteral() flags, or nested tuples thereof");
                }
            }
            @compileError("leaf index out of bounds");
        }

        /// Activity tree of the idx-th leaf of a nested query tuple.
        /// Struct leaves resolve via `componentTree`, enum leaves via
        /// `enumFlagTree`. Concrete `Error!*Tree` return, no `anytype`.
        fn queryLeafTreeAt(comptime nested: anytype, comptime idx: usize) Error!*Tree {
            comptime var cur: usize = 0;
            inline for (nested) |e| {
                if (comptime @TypeOf(e) == type) {
                    if (cur == idx) return componentTree(e);
                    cur += 1;
                } else if (comptime isEnumValue(e)) {
                    if (cur == idx) return enumFlagTree(e);
                    cur += 1;
                } else if (comptime isTuple(@TypeOf(e))) {
                    const m = comptime countQueryLeaves(e);
                    if (idx < cur + m) return queryLeafTreeAt(e, idx - cur);
                    cur += m;
                } else {
                    @compileError("query tuples hold component types, @EnumLiteral() flags, or nested tuples thereof");
                }
            }
            @compileError("leaf index out of bounds");
        }

        /// True if leaf `idx` has no equal leaf before it: dedup keeps the
        /// first occurrence, later ones are skipped on fill. Comparison is
        /// by storage key, so different enum flags (same type, different
        /// tag names) are distinct.
        pub fn isFirstQueryOccurrence(comptime nested: anytype, comptime idx: usize) bool {
            const key = comptime queryLeafKeyAt(nested, idx);
            inline for (0..idx) |j| {
                if (comptime std.mem.eql(u8, queryLeafKeyAt(nested, j), key)) return false;
            }
            return true;
        }

        /// Deduped leaf count: sizes the exact `[*Tree]` arrays in
        /// `iterateAll` after silent first-occurrence dedup.
        pub fn countUniqueQueryLeaves(comptime nested: anytype) usize {
            const n = comptime countQueryLeaves(nested);
            comptime var m: usize = 0;
            inline for (0..n) |i| {
                if (comptime isFirstQueryOccurrence(nested, i)) m += 1;
            }
            return m;
        }

        /// True if a nested query tuple contains type `C` at any depth.
        /// Mixed tuples with `@EnumLiteral()` flags are supported: enum
        /// leaves have prefixed keys that never equal `@typeName(C)`.
        pub fn queryContains(comptime nested: anytype, comptime C: type) bool {
            const want = @typeName(C);
            const n = comptime countQueryLeaves(nested);
            inline for (0..n) |i| {
                if (comptime std.mem.eql(u8, queryLeafKeyAt(nested, i), want)) return true;
            }
            return false;
        }

        /// True if a nested query tuple contains `@EnumLiteral()` flag
        /// `flag` at any depth. Parallel API to `queryContains`, which
        /// stays `type`-only: `queryContains(g, Pos)` vs
        /// `queryContainsFlag(g, .my_flag)`.
        pub fn queryContainsFlag(comptime nested: anytype, comptime flag: @EnumLiteral()) bool {
            const want = comptime enumFlagKey(flag);
            const n = comptime countQueryLeaves(nested);
            inline for (0..n) |i| {
                if (comptime std.mem.eql(u8, queryLeafKeyAt(nested, i), want)) return true;
            }
            return false;
        }

        /// Leaf shape check over nested groups: every leaf must pass
        /// `checkQueryType`; groups recurse, anything else is an error.
        /// Struct types and `@EnumLiteral()` flags are both accepted.
        fn checkQueryNested(comptime nested: anytype) void {
            inline for (nested) |e| {
                if (comptime @TypeOf(e) == type) {
                    checkQueryType(e);
                } else if (comptime isEnumValue(e)) {
                    checkQueryFlag(e);
                } else if (comptime isTuple(@TypeOf(e))) {
                    checkQueryNested(e);
                } else {
                    @compileError("query tuples hold component types, @EnumLiteral() flags, or nested tuples thereof");
                }
            }
        }

        /// Number of leaf values in a (possibly nested) create tuple type,
        /// depth-first, empty groups contribute zero.
        pub fn countValueLeaves(comptime TupleType: type) usize {
            comptime var n: usize = 0;
            const info = @typeInfo(TupleType).@"struct";
            inline for (info.field_names, info.field_types) |_, FieldType| {
                if (comptime isTuple(FieldType)) {
                    n += comptime countValueLeaves(FieldType);
                } else {
                    n += 1;
                }
            }
            return n;
        }

        /// idx-th leaf TYPE of a nested value-tuple type, depth-first.
        /// Types only, values stay in place: validation never copies
        /// payloads, the runtime walk reads leaves from nested slots.
        pub fn valueLeafTypeAt(comptime TupleType: type, comptime idx: usize) type {
            comptime var cur: usize = 0;
            const info = @typeInfo(TupleType).@"struct";
            inline for (info.field_names, info.field_types) |_, FieldType| {
                if (comptime isTuple(FieldType)) {
                    const m = comptime countValueLeaves(FieldType);
                    if (idx < cur + m) return comptime valueLeafTypeAt(FieldType, idx - cur);
                    cur += m;
                } else {
                    if (cur == idx) return FieldType;
                    cur += 1;
                }
            }
            @compileError("leaf index out of bounds");
        }

        /// Registers a component type. Creates one column sized to the
        /// current row count, all flags inactive, payload bytes zeroed.
        /// Returns `true` if the component was registered, `false` if it
        /// was already registered (idempotent no-op).
        pub fn addComponent(alloc: Allocator, comptime T: type) Error!bool {
            if (!initialized) return Error.NotInitialized;
            if (hasFields(T)) {
                if (data_map.contains(@typeName(T))) return false;
                const rows: usize = destroyed.bitset.bits_count;
                try data_cols.append(alloc, DataColumn{
                    .name = try alloc.dupe(u8, @typeName(T)),
                    .size = @sizeOf(T),
                    .stride = std.mem.alignForward(usize, @sizeOf(T), @alignOf(T)),
                    .alignment = @alignOf(T),
                });
                errdefer {
                    var leaked = data_cols.pop() orelse unreachable;
                    freeDataColumn(alloc, &leaked);
                }
                const col = &data_cols.items[data_cols.items.len - 1];
                try col.tree.resize(alloc, @intCast(rows), .inactive);
                try dataEnsureRows(alloc, col, rows);
                try data_map.put(alloc, col.name, @intCast(data_cols.items.len - 1));
            } else {
                if (flag_map.contains(@typeName(T))) return false;
                const rows: usize = destroyed.bitset.bits_count;
                try flag_cols.append(alloc, FlagColumn{
                    .name = try alloc.dupe(u8, @typeName(T)),
                });
                errdefer {
                    var leaked = flag_cols.pop() orelse unreachable;
                    leaked.tree.deinit(alloc);
                    alloc.free(leaked.name);
                }
                const col = &flag_cols.items[flag_cols.items.len - 1];
                try col.tree.resize(alloc, @intCast(rows), .inactive);
                try flag_map.put(alloc, col.name, @intCast(flag_cols.items.len - 1));
            }
            return true;
        }

        /// Unregisters a component type. Removes its column with swap-remove
        /// and repoints the map entry of the moved column by its stored name.
        /// Returns `true` if the component was removed, `false` if it was
        /// not registered (idempotent no-op).
        pub fn removeComponent(alloc: Allocator, comptime T: type) Error!bool {
            if (!initialized) return Error.NotInitialized;
            if (hasFields(T)) {
                const idx = data_map.get(@typeName(T)) orelse return false;
                _ = data_map.remove(@typeName(T));
                var gone = data_cols.swapRemove(idx);
                if (idx < data_cols.items.len) {
                    try data_map.put(alloc, data_cols.items[idx].name, idx);
                }
                freeDataColumn(alloc, &gone);
            } else {
                const idx = flag_map.get(@typeName(T)) orelse return false;
                _ = flag_map.remove(@typeName(T));
                var gone = flag_cols.swapRemove(idx);
                if (idx < flag_cols.items.len) {
                    try flag_map.put(alloc, flag_cols.items[idx].name, idx);
                }
                gone.tree.deinit(alloc);
                alloc.free(gone.name);
            }
            return true;
        }

        /// Checks registration in the matching map, chosen at comptime.
        pub fn containComponent(comptime T: type) bool {
            if (!initialized) return false;
            if (hasFields(T)) return data_map.contains(@typeName(T));
            return flag_map.contains(@typeName(T));
        }

        /// Registers an `@EnumLiteral()` flag. Creates one flag column
        /// sized to the current row count, all flags inactive.
        /// Parallel API to `addComponent`, which stays `type`-only:
        /// `addComponent(Pos)` vs `addFlag(.my_flag)`.
        /// Returns `true` if the flag was registered, `false` if it
        /// was already registered (idempotent no-op).
        pub fn addFlag(alloc: Allocator, comptime flag: @EnumLiteral()) Error!bool {
            if (!initialized) return Error.NotInitialized;
            const key = comptime enumFlagKey(flag);
            if (flag_map.contains(key)) return false;
            const rows: usize = destroyed.bitset.bits_count;
            try flag_cols.append(alloc, FlagColumn{
                .name = try alloc.dupe(u8, key),
            });
            errdefer {
                var leaked = flag_cols.pop() orelse unreachable;
                leaked.tree.deinit(alloc);
                alloc.free(leaked.name);
            }
            const col = &flag_cols.items[flag_cols.items.len - 1];
            try col.tree.resize(alloc, @intCast(rows), .inactive);
            try flag_map.put(alloc, col.name, @intCast(flag_cols.items.len - 1));
            return true;
        }

        /// Unregisters an `@EnumLiteral()` flag. Removes its column with
        /// swap-remove and repoints the map entry of the moved column.
        /// Parallel API to `removeComponent`.
        /// Returns `true` if the flag was removed, `false` if it was
        /// not registered (idempotent no-op).
        pub fn removeFlag(alloc: Allocator, comptime flag: @EnumLiteral()) Error!bool {
            if (!initialized) return Error.NotInitialized;
            const key = comptime enumFlagKey(flag);
            const idx = flag_map.get(key) orelse return false;
            _ = flag_map.remove(key);
            var gone = flag_cols.swapRemove(idx);
            if (idx < flag_cols.items.len) {
                try flag_map.put(alloc, flag_cols.items[idx].name, idx);
            }
            gone.tree.deinit(alloc);
            alloc.free(gone.name);
            return true;
        }

        /// Checks registration of an `@EnumLiteral()` flag.
        /// Parallel API to `containComponent`.
        pub fn containFlag(comptime flag: @EnumLiteral()) bool {
            if (!initialized) return false;
            return flag_map.contains(comptime enumFlagKey(flag));
        }

        /// Grows a payload buffer to `rows`, zeroing fresh bytes. The base
        /// is aligned manually inside an over-allocated `raw` slice, so any
        /// runtime `@alignOf(T)` works without comptime tricks.
        fn dataEnsureRows(alloc: Allocator, col: *DataColumn, rows: usize) Error!void {
            if (col.cap >= rows) {
                if (rows > col.len) {
                    @memset(col.buf[col.len * col.stride .. rows * col.stride], 0);
                }
                col.len = rows;
                return;
            }
            const new_cap = @max(rows, if (col.cap == 0) @as(usize, 8) else col.cap * 2);
            const raw = try alloc.alloc(u8, new_cap * col.stride + col.alignment - 1);
            errdefer alloc.free(raw);
            const off = std.mem.alignForward(usize, @intFromPtr(raw.ptr), col.alignment) - @intFromPtr(raw.ptr);
            const buf = raw[off .. off + new_cap * col.stride];
            if (col.len > 0) {
                @memcpy(buf[0 .. col.len * col.stride], col.buf[0 .. col.len * col.stride]);
            }
            @memset(buf[col.len * col.stride ..], 0);
            if (col.cap > 0) alloc.free(col.raw);
            col.raw = raw;
            col.buf = buf;
            col.cap = new_cap;
            col.len = rows;
        }

        /// Releases one payload column: tree, bytes and owned name.
        fn freeDataColumn(alloc: Allocator, col: *DataColumn) void {
            col.tree.deinit(alloc);
            if (col.cap > 0) alloc.free(col.raw);
            alloc.free(col.name);
            col.* = .{ .name = &.{}, .size = 0, .stride = 0, .alignment = 1 };
        }

        /// Number of entity rows, including destroyed ones awaiting reclaim.
        pub fn rowCount() u32 {
            if (!initialized) return 0;
            return destroyed.bitset.bits_count;
        }

        /// Number of issued slots. Grows monotonically until `deinit`.
        pub fn slotCount() u32 {
            if (!initialized) return 0;
            return @intCast(indirection.items.len);
        }

        /// Creates one entity from a tuple of component values.
        /// An empty tuple creates a bare entity. Tuples may nest: groups
        /// like `const g = .{ Transform{...}, Tag{} };` expand depth-first,
        /// e.g. `create(alloc, .{ g, Health{...} })`; empty groups vanish.
        /// Duplicate data types are comptime errors (ambiguous payload);
        /// duplicate flags dedup silently. Every listed type must be
        /// registered; listed data columns start active with copied
        /// payloads, listed flag columns start active with no payload,
        /// every other column starts inactive on the new row.
        /// `values` may be a tuple or a single component value:
        /// `create(alloc, .{ Transform{ .scale = ... } })` or
        /// `create(alloc, Transform{ .scale = ... })`, likewise
        /// `create(alloc, .{ Tag{} })` or `create(alloc, Tag{})`,
        /// and the same with `@EnumLiteral()` flags:
        /// `create(alloc, .{ .my_flag })` or `create(alloc, .my_flag)`.
        /// Tuples may mix struct values and enum flags at any depth.
        pub fn create(alloc: Allocator, values: anytype) Error!EntityReference {
            const V = @TypeOf(values);
            const ti = @typeInfo(V);
            if (ti == .@"struct" and ti.@"struct".is_tuple) {
                return createChecked(alloc, values);
            } else if (ti == .@"struct") {
                return createChecked(alloc, .{values});
            } else if (ti == .enum_literal) {
                return createChecked(alloc, .{values});
            } else {
                @compileError("ECSTable.create(alloc, values): `values` must be a tuple of component values or a single component value, e.g. `.{ Transform{ .scale = ... } }` or `Transform{ .scale = ... }`; got `" ++ @typeName(V) ++ "`.");
            }
        }

        /// Checked body of `create`: `values` are known to be a tuple here,
        /// so the `inline for` loops below never see a mistyped argument and
        /// no cascading errors are reported after the gate above.
        fn createChecked(alloc: Allocator, values: anytype) Error!EntityReference {
            if (!initialized) return Error.NotInitialized;
            validateTuple(values);
            try ensureRegistered(values);
            const row = try allocRow(alloc);
            for (data_cols.items) |*col| col.tree.setBit(row, .inactive);
            for (flag_cols.items) |*col| col.tree.setBit(row, .inactive);
            try activateRowValues(row, values);
            const slot: u32 = @intCast(indirection.items.len);
            try indirection.append(alloc, row);
            try generations.append(alloc, 0);
            if (cleared.bitset.bits_count <= slot) {
                try cleared.resize(alloc, slot + 64, .inactive);
            }
            destroyed.setBit(row, .inactive);
            activity.setBit(row, .active);
            row_to_slot.items[row] = slot;
            return .{ .slot = slot, .gen = 0 };
        }

        /// Creates `n` entities from the same tuple, invoking `cb` with
        /// each fresh handle. Handles stream through the comptime callback
        /// one by one, no handle array is ever allocated.
        /// `values` may be a tuple or a single component value; nested
        /// groups are supported exactly like in `create`.
        pub fn createN(alloc: Allocator, values: anytype, n: u32, context: anytype, comptime cb: fn (@TypeOf(context), EntityReference) void) Error!void {
            const V = @TypeOf(values);
            const ti = @typeInfo(V);
            if (ti != .@"struct" and ti != .enum_literal) {
                @compileError("ECSTable.createN(alloc, values, n, ...): `values` must be a tuple of component values or a single component value, e.g. `.{ Transform{ .scale = ... } }` or `Transform{ .scale = ... }`; got `" ++ @typeName(V) ++ "`.");
            } else {
                if (!initialized) return Error.NotInitialized;
                var i: u32 = 0;
                while (i < n) : (i += 1) {
                    cb(context, try create(alloc, values));
                }
            }
        }

        /// Reclaims every destroyed row. Live tail donors swap-remove into
        /// head victims via backward iteration; victim slots are sealed in
        /// `cleared` forever; every row plane is truncated to the live count
        /// while slot storage only ever grows.
        pub fn clearDestroyed(alloc: Allocator) Error!void {
            if (!initialized) return Error.NotInitialized;
            const total = destroyed.bitset.bits_count;
            var victims: std.ArrayListUnmanaged(u32) = .empty;
            defer victims.deinit(alloc);
            {
                var ctx = VictimCtx{ .list = &victims, .alloc = alloc };
                const It = Tree.Iterator(*VictimCtx, VictimCtx.push, null, .forward);
                // Victim/Donor push callbacks never fail: OOM is reported
                // via the ctx flag, so unwrapping here is safe.
                _ = It.iterateAll(.{ .tree = &destroyed, .context = &ctx }, null, null) catch unreachable;
                if (ctx.oom) return Error.OutOfMemory;
            }
            if (victims.items.len == 0) return;
            const live: u32 = total - @as(u32, @intCast(victims.items.len));

            var donors: std.ArrayListUnmanaged(u32) = .empty;
            defer donors.deinit(alloc);
            var need: usize = 0;
            for (victims.items) |v| {
                if (v < live) need += 1;
            }
            if (need > 0) {
                var ctx = DonorCtx{ .list = &donors, .alloc = alloc, .need = need };
                const It = Tree.Iterator(*DonorCtx, null, DonorCtx.push, .backward);
                // Donor push never fails (OOM via ctx flag), unwrapping is safe.
                _ = It.iterateAll(.{ .tree = &destroyed, .context = &ctx }, live, null) catch unreachable;
                if (ctx.oom) return Error.OutOfMemory;
                std.debug.assert(donors.items.len == need);
            }

            var di: usize = 0;
            for (victims.items) |v| {
                if (v < live) {
                    const d = donors.items[di];
                    di += 1;
                    const s_v = row_to_slot.items[v];
                    swapRow(v, d);
                    sealSlot(s_v);
                } else {
                    sealSlot(row_to_slot.items[v]);
                }
            }

            try destroyed.resize(alloc, live, .inactive);
            try activity.resize(alloc, live, .inactive);
            for (data_cols.items) |*col| {
                try col.tree.resize(alloc, live, .inactive);
                col.len = live;
            }
            for (flag_cols.items) |*col| {
                try col.tree.resize(alloc, live, .inactive);
            }
            row_to_slot.shrinkRetainingCapacity(live);
        }

        /// Component query over activity flags, thin wrapper over the tree
        /// common iterator. The single callback fires for the active match:
        /// rows with every `Includes` component enabled minus rows with any
        /// `Excludes` component enabled.
        ///
        /// `Includes`/`Excludes` accept nested groups: tuples may contain
        /// component types, `@EnumLiteral()` flags, or other tuples, e.g.
        /// `const g = .{ A, .my_flag };` then `Query(.{ g, C }, ...)`.
        /// A single component type or flag may pass bare, without a tuple:
        /// `Query(A, .{})` means the same as `Query(.{A}, .{})`, and
        /// `Query(.my_flag, .{})` means the same as `Query(.{.my_flag}, .{})`.
        /// Groups expand depth-first into one flat list, empty groups vanish,
        /// duplicate leaves dedup silently (struct types by `==`, enum flags
        /// by tag name). A leaf listed on both sides is a comptime error.
        ///
        /// Idioms, no second callback needed:
        /// - entities with A and B but not C: Includes={A,B}, Excludes={C}.
        /// - every entity except ones having C: Includes={}, Excludes={C}.
        /// - entities with C disabled: Includes={}, Excludes={C}. The active
        ///   mask is the universe minus C-active rows, exactly the C-inactive
        ///   set, so the disabled case is an Excludes query, not a mode.
        /// - entities with C enabled: Includes={C}, Excludes={}.
        ///
        /// Implicit filters, same for every query: `destroyed` rows never
        /// visit; `entity_activity` selects the entity flag when non-null.
        /// Empty user Includes degrades to a full live-row scan with filters;
        /// a fully empty scan is impossible: `destroyed` always vetoes.
        /// Inside callbacks only `Row` methods are allowed: rows are unstable
        /// across `clearDestroyed`, and handle resolution per row is wasteful.
        /// To name a visited row afterwards, call `rowToEntity` explicitly.
        pub fn Query(
            comptime Includes: anytype,
            comptime Excludes: anytype,
            comptime direction: bit_tree.Direction,
            comptime Context: type,
            comptime on_row: fn (ctx: Context, row: u32) callconv(.@"inline") anyerror!bool,
        ) type {
            comptime validateQuery(Includes, Excludes);
            const NormI = if (comptime isTuple(@TypeOf(Includes))) Includes else .{Includes};
            const NormE = if (comptime isTuple(@TypeOf(Excludes))) Excludes else .{Excludes};
            comptime checkQueryNested(NormI);
            comptime checkQueryNested(NormE);
            comptime validateQueryCross(NormI, NormE);
            const IL = countUniqueQueryLeaves(NormI);
            const EL = countUniqueQueryLeaves(NormE);
            const NI = countQueryLeaves(NormI);
            const NE = countQueryLeaves(NormE);
            return struct {
                /// Runs the match over an optional bit range with an optional
                /// entity activity filter. Returns false on early callback
                /// exit (`on_row` returned false); a `try` inside `on_row`
                /// aborts the walk with that error instead.
                pub fn iterateAll(ctx: Context, start_bit: ?u32, end_bit: ?u32, entity_activity: ?bool) anyerror!bool {
                    if (!initialized) return Error.NotInitialized;
                    var inc: [IL]*Tree = undefined;
                    {
                        var cursor: usize = 0;
                        inline for (0..NI) |i| {
                            if (comptime isFirstQueryOccurrence(NormI, i)) {
                                inc[cursor] = try Table.queryLeafTreeAt(NormI, i);
                                cursor += 1;
                            }
                        }
                        std.debug.assert(cursor == IL);
                    }
                    var exc: [EL + 1]*Tree = undefined;
                    {
                        var cursor: usize = 0;
                        inline for (0..NE) |i| {
                            if (comptime isFirstQueryOccurrence(NormE, i)) {
                                exc[cursor] = try Table.queryLeafTreeAt(NormE, i);
                                cursor += 1;
                            }
                        }
                        std.debug.assert(cursor == EL);
                    }
                    exc[EL] = &destroyed;
                    if (entity_activity) |ea| {
                        if (ea) {
                            var inc_w: [IL + 1]*Tree = undefined;
                            inline for (0..IL) |k| inc_w[k] = inc[k];
                            inc_w[IL] = &activity;
                            const It = Tree.CommonIterator(IL + 1, EL + 1, Context, on_row, null, direction);
                            return It.iterateAll(.{ .includes = inc_w, .excludes = exc, .context = ctx }, start_bit, end_bit);
                        } else {
                            var exc_w: [EL + 2]*Tree = undefined;
                            inline for (0..EL + 1) |k| exc_w[k] = exc[k];
                            exc_w[EL + 1] = &activity;
                            const It = Tree.CommonIterator(IL, EL + 2, Context, on_row, null, direction);
                            return It.iterateAll(.{ .includes = inc, .excludes = exc_w, .context = ctx }, start_bit, end_bit);
                        }
                    } else {
                        const It = Tree.CommonIterator(IL, EL + 1, Context, on_row, null, direction);
                        return It.iterateAll(.{ .includes = inc, .excludes = exc, .context = ctx }, start_bit, end_bit);
                    }
                }
            };
        }

        /// Comptime shape check of a query: each side is a tuple, a single
        /// component type, or a nested group. Leaf shape is checked by
        /// `checkQueryNested`, cross-side conflicts by `validateQueryCross`.
        fn validateQuery(Includes: anytype, Excludes: anytype) void {
            validateQuerySide(Includes, "Includes");
            validateQuerySide(Excludes, "Excludes");
        }

        /// One side of a query: tuple/group passes through, a bare type or
        /// bare `@EnumLiteral()` flag is wrapped into a 1-tuple downstream;
        /// anything else (e.g. a component value or a number) is a comptime
        /// error.
        fn validateQuerySide(S: anytype, comptime side: []const u8) void {
            const T = @TypeOf(S);
            if (comptime isTuple(T)) return;
            if (comptime T == type) return;
            if (comptime @typeInfo(T) == .enum_literal) return;
            if (comptime @typeInfo(T) == .@"struct") @compileError("ECSTable.Query(...): `" ++ side ++ "` takes component TYPES, not values; pass `" ++ @typeName(T) ++ "` instead of `" ++ @typeName(T) ++ "{ ... }`.");
            @compileError("ECSTable.Query(...): `" ++ side ++ "` must be a tuple of component types, @EnumLiteral() flags, a single component type/flag, or a nested group; got `" ++ @typeName(T) ++ "`.");
        }

        /// Cross-side check over nested groups: any leaf present on both
        /// sides is a comptime error, compared by storage key so struct
        /// types and enum flags (same type, different tag names) are
        /// distinguished. Duplicates inside each side are deduped
        /// silently on fill instead.
        fn validateQueryCross(Inc: anytype, Exc: anytype) void {
            const ni = comptime countQueryLeaves(Inc);
            const ne = comptime countQueryLeaves(Exc);
            inline for (0..ni) |i| {
                const key = comptime queryLeafKeyAt(Inc, i);
                inline for (0..ne) |j| {
                    if (comptime std.mem.eql(u8, queryLeafKeyAt(Exc, j), key)) @compileError("component in both Includes and Excludes");
                }
            }
        }

        /// Query tuple elements must be struct types, values are rejected.
        /// Kept `type`-only; `@EnumLiteral()` leaves go via `checkQueryFlag`.
        fn checkQueryType(X: anytype) void {
            if (@TypeOf(X) != type) @compileError("query tuples hold component types, not values");
            if (@typeInfo(X) != .@"struct") @compileError("component must be a struct type");
        }

        /// Query tuple `@EnumLiteral()` leaves are always well-formed:
        /// identity is the tag name, registration is checked at
        /// `iterateAll` time via `ComponentNotFound`.
        fn checkQueryFlag(flag: @EnumLiteral()) void {
            _ = flag;
        }

        /// Dead slot marker: sealed slots never point at rows again.
        const dead_slot: u32 = std.math.maxInt(u32);

        /// Copies one live donor row over a victim row across every plane,
        /// payload bytes and the reverse index. Victim slot sealed by caller.
        fn swapRow(v: u32, d: u32) void {
            copyBit(&destroyed, v, d);
            copyBit(&activity, v, d);
            for (data_cols.items) |*col| {
                copyBit(&col.tree, v, d);
                @memcpy(col.buf[@as(usize, v) * col.stride ..][0..col.size], col.buf[@as(usize, d) * col.stride ..][0..col.size]);
            }
            for (flag_cols.items) |*col| copyBit(&col.tree, v, d);
            const s_d = row_to_slot.items[d];
            row_to_slot.items[v] = s_d;
            indirection.items[s_d] = v;
        }

        /// Copies one flag lane between rows of the same tree.
        fn copyBit(tree: *Tree, dst: u32, src: u32) void {
            tree.setBit(dst, tree.bitset.getBit(src));
        }

        /// Seals one victim slot: marks cleared and unpoints indirection.
        /// Cleared storage always covers issued slots: it grows in chunks
        /// on every `create` and is never truncated afterwards.
        fn sealSlot(s_v: u32) void {
            cleared.setBit(s_v, .active);
            indirection.items[s_v] = dead_slot;
        }

        /// Comptime shape check of a create tuple: struct values (payload
        /// or flag) and `@EnumLiteral()` flags, nested groups expanded via
        /// the `countValueLeaves`/`valueLeafTypeAt` flat view. Duplicate
        /// data types are comptime errors; duplicate flags of either kind
        /// are allowed (idempotent setBit). All enum literals share one
        /// type, so enum leaves are skipped here: any number of them is
        /// legal, dedup happens silently at activation.
        fn validateTuple(values: anytype) void {
            const ti = @typeInfo(@TypeOf(values));
            if (ti != .@"struct" or !ti.@"struct".is_tuple) @compileError("create expects a tuple of component values");
            validateTupleLeaves(@TypeOf(values));
        }

        /// Leaf validation on a genuine comptime tuple type: with `TT` as a
        /// `comptime` param every derived count stays comptime-known, which
        /// a runtime `values: anytype` param cannot guarantee.
        fn validateTupleLeaves(comptime TT: type) void {
            const N = comptime countValueLeaves(TT);
            inline for (0..N) |k| {
                const F = comptime valueLeafTypeAt(TT, k);
                if (comptime @typeInfo(F) == .enum_literal) continue;
                if (@typeInfo(F) != .@"struct") @compileError("tuple element must be a component struct value or @EnumLiteral() flag");
                if (comptime isTuple(F)) @compileError("tuple element must be a component struct value or @EnumLiteral() flag");
                inline for (0..k) |j| {
                    const G = comptime valueLeafTypeAt(TT, j);
                    if (comptime @typeInfo(G) == .enum_literal) continue;
                    if (comptime G == F) {
                        if (comptime hasFields(F)) @compileError("duplicate component type in create tuple");
                    }
                }
            }
        }

        /// Registration check over nested groups: `isTuple` first so empty
        /// groups vanish instead of hitting the flag path. Struct leaves
        /// use `containComponent`, `@EnumLiteral()` leaves use
        /// `containFlag`. Leaves keep the same `ComponentNotFound`
        /// contract as flat tuples.
        fn ensureRegistered(values: anytype) Error!void {
            inline for (values) |v| {
                if (comptime isTuple(@TypeOf(v))) {
                    try ensureRegistered(v);
                } else if (comptime isEnumType(@TypeOf(v))) {
                    if (!containFlag(v)) return Error.ComponentNotFound;
                } else {
                    if (!containComponent(@TypeOf(v))) return Error.ComponentNotFound;
                }
            }
        }

        /// Payload copy over nested groups without materializing a flat
        /// tuple: leaves are copied straight from their nested slots into
        /// the column, one `memcpy` per data leaf, `setBit` per flag leaf
        /// (struct flags and `@EnumLiteral()` flags alike).
        fn activateRowValues(row: u32, values: anytype) Error!void {
            inline for (values) |v| {
                if (comptime isTuple(@TypeOf(v))) {
                    try activateRowValues(row, v);
                } else if (comptime isEnumType(@TypeOf(v))) {
                    const tree = try enumFlagTree(v);
                    tree.setBit(row, .active);
                } else if (comptime hasFields(@TypeOf(v))) {
                    const col = try dataColumn(@TypeOf(v));
                    col.tree.setBit(row, .active);
                    @memcpy(rowBytes(col, row), std.mem.asBytes(&v));
                } else {
                    const tree = try componentTree(@TypeOf(v));
                    tree.setBit(row, .active);
                }
            }
        }

        /// Collects destroyed row ids in forward order.
        const VictimCtx = struct {
            list: *std.ArrayListUnmanaged(u32),
            alloc: Allocator,
            oom: bool = false,

            inline fn push(ctx: *VictimCtx, id: u32) anyerror!bool {
                ctx.list.append(ctx.alloc, id) catch {
                    ctx.oom = true;
                    return false;
                };
                return true;
            }
        };

        /// Collects live donor rows from the tail in backward order,
        /// stopping once `need` donors are gathered.
        const DonorCtx = struct {
            list: *std.ArrayListUnmanaged(u32),
            alloc: Allocator,
            need: usize,
            oom: bool = false,

            inline fn push(ctx: *DonorCtx, id: u32) anyerror!bool {
                ctx.list.append(ctx.alloc, id) catch {
                    ctx.oom = true;
                    return false;
                };
                return ctx.list.items.len < ctx.need;
            }
        };

        /// Resolves a live row back into a stable handle. Explicit
        /// conversion for iteration callbacks that only see row ids.
        pub fn rowToEntity(row: u32) Error!EntityReference {
            if (!initialized) return Error.NotInitialized;
            if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
            if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
            const slot = row_to_slot.items[row];
            if (slot >= indirection.items.len) return Error.InvalidReference;
            return .{ .slot = slot, .gen = generations.items[slot] };
        }

        /// Validates a handle: slot bounds, generation match, slot not
        /// cleared, target row inside bounds and not destroyed.
        fn isValidRef(ref: EntityReference) bool {
            if (!initialized) return false;
            if (ref.slot >= indirection.items.len) return false;
            if (generations.items[ref.slot] != ref.gen) return false;
            if (ref.slot < cleared.bitset.bits_count and cleared.bitset.getBit(ref.slot) == .active) return false;
            const row = indirection.items[ref.slot];
            if (row >= destroyed.bitset.bits_count) return false;
            if (destroyed.bitset.getBit(row) == .active) return false;
            return true;
        }

        /// Maps a validated handle to its current row.
        fn resolveRef(ref: EntityReference) Error!u32 {
            if (!isValidRef(ref)) return Error.InvalidReference;
            return indirection.items[ref.slot];
        }

        /// Marks the row destroyed and bumps its slot generation so
        /// outstanding handles fail validation immediately.
        fn destroyRow(row: u32) void {
            destroyed.setBit(row, .active);
            const slot = row_to_slot.items[row];
            generations.items[slot] +%= 1;
        }

        /// Finds the lowest destroyed row or grows every row plane by one,
        /// component trees and payload buffers included.
        fn allocRow(alloc: Allocator) Error!u32 {
            var finder = ReuseCtx{};
            const It = Tree.Iterator(*ReuseCtx, ReuseCtx.push, null, .forward);
            // ReuseCtx.push never fails (pure early-exit scan), so unwrapping is safe.
            _ = It.iterateAll(.{ .tree = &destroyed, .context = &finder }, null, null) catch unreachable;
            if (finder.row) |row| return row;
            const row = destroyed.bitset.bits_count;
            try destroyed.resize(alloc, row + 1, .inactive);
            try activity.resize(alloc, row + 1, .inactive);
            try row_to_slot.append(alloc, 0);
            for (data_cols.items) |*col| {
                try col.tree.resize(alloc, row + 1, .inactive);
                try dataEnsureRows(alloc, col, row + 1);
            }
            for (flag_cols.items) |*col| {
                try col.tree.resize(alloc, row + 1, .inactive);
            }
            return row;
        }

        /// Early-exit context stopping the destroyed scan at the first hit.
        const ReuseCtx = struct {
            row: ?u32 = null,

            inline fn push(ctx: *ReuseCtx, id: u32) anyerror!bool {
                ctx.row = id;
                return false;
            }
        };
    };
}

const t = std.testing;

test "ECSTable core: create/isValid/destroy lifecycle" {
    const T = ECSTable(.t101);
    try T.init();
    defer T.deinit(t.allocator);

    const a = try T.create(t.allocator, .{});
    const b = try T.create(t.allocator, .{});
    try t.expect(a.isValid());
    try t.expect(b.isValid());
    try t.expect(a.slot != b.slot);
    try t.expectEqual(@as(u32, 2), T.rowCount());
    try t.expectEqual(@as(u32, 2), T.slotCount());
    try t.expect(try a.isActive());

    try a.destroy();
    try t.expect(!a.isValid());
    try t.expect(b.isValid());
    try t.expectError(T.Error.InvalidReference, a.isActive());
    try t.expectError(T.Error.InvalidReference, a.destroy());
    try t.expectError(T.Error.AlreadyDestroyed, T.rowToEntity(0));

    const bad = T.EntityReference{ .slot = 999, .gen = 0 };
    try t.expect(!bad.isValid());
    try t.expectError(T.Error.InvalidReference, bad.destroy());
}

test "ECSTable core: destroyed rows are reused lowest-first" {
    const T = ECSTable(.t102);
    try T.init();
    defer T.deinit(t.allocator);

    const a = try T.create(t.allocator, .{});
    const b = try T.create(t.allocator, .{});
    const c = try T.create(t.allocator, .{});
    try b.destroy();
    try t.expectEqual(@as(u32, 3), T.rowCount());

    const d = try T.create(t.allocator, .{});
    try t.expectEqual(@as(u32, 3), T.rowCount());
    try t.expectEqual(@as(u32, 4), T.slotCount());
    const rd = try T.rowToEntity(1);
    try t.expectEqual(d.slot, rd.slot);
    try t.expectEqual(d.gen, rd.gen);
    try t.expect(d.isValid());
    _ = a;
    _ = c;
}

test "ECSTable core: createN streams handles through the callback" {
    const T = ECSTable(.t103);
    try T.init();
    defer T.deinit(t.allocator);

    const Ctx = struct {
        count: u32 = 0,
        last: T.EntityReference = .{ .slot = 0, .gen = 0 },

        fn push(self: *@This(), ref: T.EntityReference) void {
            self.count += 1;
            self.last = ref;
            std.debug.assert(ref.isValid());
        }
    };
    var ctx = Ctx{};
    try T.createN(t.allocator, .{}, 5, &ctx, Ctx.push);
    try t.expectEqual(@as(u32, 5), ctx.count);
    try t.expectEqual(@as(u32, 5), T.rowCount());
    try t.expect(ctx.last.isValid());
}

test "ECSTable core: Row plane validates bounds and liveness" {
    const T = ECSTable(.t104);
    try T.init();
    defer T.deinit(t.allocator);

    const a = try T.create(t.allocator, .{});
    try t.expect(T.Row.isAlive(0));
    try t.expect(!T.Row.isAlive(7));
    try t.expect(try T.Row.isActive(0));
    try T.Row.setActivity(0, .inactive);
    try t.expect(!try T.Row.isActive(0));
    try t.expect(!try a.isActive());
    try a.setActivity(.active);
    try t.expect(try a.isActive());

    try T.Row.destroy(0);
    try t.expect(!T.Row.isAlive(0));
    try t.expectError(T.Error.AlreadyDestroyed, T.Row.destroy(0));
    try t.expectError(T.Error.RowOutOfBounds, T.Row.isActive(9));
    try t.expectError(T.Error.RowOutOfBounds, T.Row.setActivity(9, .active));
}

test "ECSTable core: rowToEntity tracks moves of the same row" {
    const T = ECSTable(.t105);
    try T.init();
    defer T.deinit(t.allocator);

    const a = try T.create(t.allocator, .{});
    const b = try T.create(t.allocator, .{});
    const ra = try T.rowToEntity(0);
    const rb = try T.rowToEntity(1);
    try t.expectEqual(a.slot, ra.slot);
    try t.expectEqual(b.slot, rb.slot);
    try t.expectError(T.Error.RowOutOfBounds, T.rowToEntity(2));
}

const Pos = struct { x: f32, y: f32 };
const Vel = struct { dx: f32, dy: f32 };
const Health = struct { hp: u32 };
const Tag = struct {};
const Ghost = struct { v: u8 };

test "ECSTable columns: add/remove/contain with swap remap" {
    const T = ECSTable(.t201);
    try T.init();
    defer T.deinit(t.allocator);

    try t.expect(!T.containComponent(Pos));
    try t.expect(try T.addComponent(t.allocator, Pos));
    try t.expect(try T.addComponent(t.allocator, Vel));
    try t.expect(try T.addComponent(t.allocator, Tag));
    try t.expect(T.containComponent(Pos));
    try t.expect(T.containComponent(Vel));
    try t.expect(T.containComponent(Tag));
    try t.expect(!T.containComponent(Health));
    try t.expect(!try T.addComponent(t.allocator, Pos));
    try t.expect(!try T.addComponent(t.allocator, Tag));

    const e = try T.create(t.allocator, .{ Pos{ .x = 1, .y = 2 }, Vel{ .dx = 3, .dy = 4 } });
    try t.expectEqual(Pos{ .x = 1, .y = 2 }, try e.getComponent(Pos));

    try t.expect(try T.removeComponent(t.allocator, Pos));
    try t.expect(!T.containComponent(Pos));
    try t.expect(T.containComponent(Vel));
    try t.expect(!try T.removeComponent(t.allocator, Pos));
    try t.expectError(T.Error.ComponentNotFound, e.getComponent(Pos));
    try t.expectEqual(Vel{ .dx = 3, .dy = 4 }, try e.getComponent(Vel));

    try t.expect(try T.removeComponent(t.allocator, Tag));
    try t.expect(!T.containComponent(Tag));
    try t.expect(try T.addComponent(t.allocator, Pos));
    try t.expect(T.containComponent(Pos));
}

test "ECSTable columns: late add sizes rows, get/set/ptr roundtrip" {
    const T = ECSTable(.t202);
    try T.init();
    defer T.deinit(t.allocator);

    const a = try T.create(t.allocator, .{});
    const b = try T.create(t.allocator, .{});
    const c = try T.create(t.allocator, .{});
    _ = try T.addComponent(t.allocator, Health);
    try t.expectEqual(@as(u32, 3), T.rowCount());

    try a.setComponent(Health{ .hp = 10 });
    try b.setComponent(Health{ .hp = 20 });
    try c.setComponent(Health{ .hp = 30 });
    try t.expectEqual(@as(u32, 10), (try a.getComponent(Health)).hp);
    try t.expectEqual(@as(u32, 30), (try T.Row.getComponent(2, Health)).hp);

    const p = try b.getComponentPtr(Health);
    p.hp = 25;
    try t.expectEqual(@as(u32, 25), (try b.getComponent(Health)).hp);

    try T.Row.setComponent(0, Health{ .hp = 11 });
    try t.expectEqual(@as(u32, 11), (try a.getComponent(Health)).hp);
}

test "ECSTable create tuple: payloads, flags, missing type" {
    const T = ECSTable(.t203);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    _ = try T.addComponent(t.allocator, Vel);
    _ = try T.addComponent(t.allocator, Health);
    _ = try T.addComponent(t.allocator, Tag);

    const e = try T.create(t.allocator, .{ Pos{ .x = 5, .y = 6 }, Vel{ .dx = 7, .dy = 8 } });
    try t.expect(try e.isComponentActive(Pos));
    try t.expect(try e.isComponentActive(Vel));
    try t.expect(!try e.isComponentActive(Health));
    try t.expect(!try e.isComponentActive(Tag));
    try t.expectEqual(Pos{ .x = 5, .y = 6 }, try e.getComponent(Pos));
    try t.expectEqual(Vel{ .dx = 7, .dy = 8 }, try T.Row.getComponent(0, Vel));

    try e.setComponentActivity(Health, .active);
    try t.expect(try e.isComponentActive(Health));
    try T.Row.setComponentActivity(0, Tag, .active);
    try t.expect(try T.Row.isComponentActive(0, Tag));

    try t.expectError(T.Error.ComponentNotFound, T.create(t.allocator, .{Ghost{ .v = 1 }}));
    try t.expectError(T.Error.ComponentNotFound, e.getComponent(Ghost));
    try t.expectError(T.Error.ComponentNotFound, e.isComponentActive(Ghost));
}

test "ECSTable clearDestroyed: swap-remove moves data and seals slots" {
    const T = ECSTable(.t204);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    var refs: [5]T.EntityReference = undefined;
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        refs[i] = try T.create(t.allocator, .{});
        try refs[i].setComponent(Pos{ .x = @floatFromInt(i), .y = 0 });
        try refs[i].setComponentActivity(Pos, .active);
    }
    try refs[4].setActivity(.inactive);
    try refs[1].destroy();
    try refs[3].destroy();

    try T.clearDestroyed(t.allocator);
    try t.expectEqual(@as(u32, 3), T.rowCount());
    try t.expectEqual(@as(u32, 5), T.slotCount());
    try t.expect(refs[0].isValid());
    try t.expect(!refs[1].isValid());
    try t.expect(refs[2].isValid());
    try t.expect(!refs[3].isValid());
    try t.expect(refs[4].isValid());

    try t.expectEqual(@as(f32, 0), (try refs[0].getComponent(Pos)).x);
    try t.expectEqual(@as(f32, 4), (try refs[4].getComponent(Pos)).x);
    try t.expectEqual(@as(f32, 2), (try refs[2].getComponent(Pos)).x);
    try t.expect(!try refs[4].isActive());

    const f = try T.create(t.allocator, .{});
    try t.expectEqual(@as(u32, 4), T.rowCount());
    try t.expectEqual(@as(u32, 5), f.slot);
    try t.expect(f.isValid());

    try T.clearDestroyed(t.allocator);
    try t.expectEqual(@as(u32, 4), T.rowCount());
}

const Collect = struct {
    rows: [16]u32 = undefined,
    n: usize = 0,

    inline fn push(self: *Collect, row: u32) anyerror!bool {
        self.rows[self.n] = row;
        self.n += 1;
        return true;
    }
};

test "ECSTable Query: truth table, idioms, direction, range, filters" {
    const T = ECSTable(.t205);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    _ = try T.addComponent(t.allocator, Vel);
    _ = try T.addComponent(t.allocator, Tag);

    const r0 = try T.create(t.allocator, .{Pos{ .x = 0, .y = 0 }});
    const r1 = try T.create(t.allocator, .{ Pos{ .x = 1, .y = 1 }, Vel{ .dx = 1, .dy = 1 } });
    const r2 = try T.create(t.allocator, .{Vel{ .dx = 2, .dy = 2 }});
    const r3 = try T.create(t.allocator, .{Pos{ .x = 3, .y = 3 }});
    try r3.setActivity(.inactive);
    const r4 = try T.create(t.allocator, .{});
    try r4.setComponentActivity(Tag, .active);
    const r5 = try T.create(t.allocator, .{Pos{ .x = 5, .y = 5 }});
    try r5.destroy();

    const QPos = T.Query(.{Pos}, .{}, .forward, *Collect, Collect.push);
    var c = Collect{};
    try t.expect(try QPos.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 3 }, c.rows[0..c.n]);

    const QPosNoVel = T.Query(.{Pos}, .{Vel}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try QPosNoVel.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 3 }, c.rows[0..c.n]);

    const QAll = T.Query(.{}, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try QAll.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 2, 3, 4 }, c.rows[0..c.n]);

    const QActive = T.Query(.{Pos}, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try QActive.iterateAll(&c, null, null, true));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 1 }, c.rows[0..c.n]);
    c = Collect{};
    try t.expect(try QActive.iterateAll(&c, null, null, false));
    try t.expectEqualSlices(u32, &[_]u32{3}, c.rows[0..c.n]);

    const QPosDisabled = T.Query(.{}, .{Pos}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try QPosDisabled.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 2, 4 }, c.rows[0..c.n]);

    const QTag = T.Query(.{Tag}, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try QTag.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{4}, c.rows[0..c.n]);

    const QBack = T.Query(.{Pos}, .{}, .backward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try QBack.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 3, 1, 0 }, c.rows[0..c.n]);

    c = Collect{};
    try t.expect(try QPos.iterateAll(&c, 1, 3, null));
    try t.expectEqualSlices(u32, &[_]u32{1}, c.rows[0..c.n]);
    c = Collect{};
    try t.expect(try QBack.iterateAll(&c, 3, 1, null));
    try t.expectEqualSlices(u32, &[_]u32{1}, c.rows[0..c.n]);
    c = Collect{};
    try t.expect(try QBack.iterateAll(&c, 4, 0, null));
    try t.expectEqualSlices(u32, &[_]u32{ 3, 1, 0 }, c.rows[0..c.n]);

    const QGhost = T.Query(.{Ghost}, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expectError(T.Error.ComponentNotFound, QGhost.iterateAll(&c, null, null, null));

    try t.expect(r0.isValid());
    try t.expect(r1.isValid());
    try t.expect(r2.isValid());
    try t.expect(r3.isValid());
    try t.expect(r4.isValid());
    try t.expect(!r5.isValid());
}

const FallibleCollect = struct {
    rows: [16]u32 = undefined,
    n: usize = 0,
    fail_at: usize = std.math.maxInt(usize),

    inline fn push(self: *FallibleCollect, row: u32) anyerror!bool {
        if (self.n == self.fail_at) return error.RowCallbackFailed;
        self.rows[self.n] = row;
        self.n += 1;
        return true;
    }
};

test "ECSTable Query fallible: try inside on_row aborts with that error" {
    const T = ECSTable(.t206);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    _ = try T.addComponent(t.allocator, Vel);
    _ = try T.create(t.allocator, .{Pos{ .x = 0, .y = 0 }});
    _ = try T.create(t.allocator, .{ Pos{ .x = 1, .y = 1 }, Vel{ .dx = 1, .dy = 1 } });
    _ = try T.create(t.allocator, .{Pos{ .x = 2, .y = 2 }});

    const Q = T.Query(.{Pos}, .{}, .forward, *FallibleCollect, FallibleCollect.push);

    // User error aborts the walk and propagates through iterateAll.
    var c = FallibleCollect{ .fail_at = 2 };
    try t.expectError(error.RowCallbackFailed, Q.iterateAll(&c, null, null, null));
    try t.expectEqual(@as(usize, 2), c.n);

    // Backward walk aborts the same way.
    const QB = T.Query(.{Pos}, .{}, .backward, *FallibleCollect, FallibleCollect.push);
    var cb = FallibleCollect{ .fail_at = 1 };
    try t.expectError(error.RowCallbackFailed, QB.iterateAll(&cb, null, null, null));
    try t.expectEqual(@as(usize, 1), cb.n);

    // No failure: full walk completes with true.
    var ok = FallibleCollect{};
    try t.expect(try Q.iterateAll(&ok, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, ok.rows[0..ok.n]);
}

test "ECSTable flags: create/set/get/ptr treat flag and data uniformly" {
    const T = ECSTable(.t207);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    _ = try T.addComponent(t.allocator, Tag);

    // create with mixed data + flag instance: flag only sets the bit.
    const e = try T.create(t.allocator, .{ Pos{ .x = 1, .y = 2 }, Tag{} });
    try t.expect(try e.isComponentActive(Pos));
    try t.expect(try e.isComponentActive(Tag));
    try t.expectEqual(Pos{ .x = 1, .y = 2 }, try e.getComponent(Pos));
    _ = try e.getComponent(Tag);

    // create with only a flag.
    const f = try T.create(t.allocator, .{Tag{}});
    try t.expect(try f.isComponentActive(Tag));
    try t.expect(!try f.isComponentActive(Pos));
    _ = try T.Row.getComponent(1, Tag);

    // setComponent with a flag only sets the bit.
    const g = try T.create(t.allocator, .{});
    try t.expect(!try g.isComponentActive(Tag));
    try g.setComponent(Tag{});
    try t.expect(try g.isComponentActive(Tag));
    _ = try g.getComponent(Tag);
    try T.Row.setComponent(2, Tag{});
    try t.expect(try T.Row.isComponentActive(2, Tag));

    // getComponentPtr on flags shares one address per column, writes are no-ops.
    const p0 = try e.getComponentPtr(Tag);
    const p1 = try f.getComponentPtr(Tag);
    try t.expect(p0 == p1);
    p0.* = .{};
    try t.expect(try e.isComponentActive(Tag));
    const q = try T.Row.getComponentPtr(0, Tag);
    try t.expect(q == p0);

    // Unregistered flag still reports ComponentNotFound everywhere.
    const Unreg = struct {};
    try t.expectError(T.Error.ComponentNotFound, T.create(t.allocator, .{Unreg{}}));
    try t.expectError(T.Error.ComponentNotFound, e.getComponent(Unreg));
    try t.expectError(T.Error.ComponentNotFound, e.getComponentPtr(Unreg));
    try t.expectError(T.Error.ComponentNotFound, e.isComponentActive(Unreg));
}

test "nested query helpers: flat view and dedup over 3+ levels" {
    const T = ECSTable(.t301);

    const g1 = .{ Pos, Vel };
    const g2 = .{ g1, Health, .{} };
    const g3 = .{ g2, Tag };
    // Flat view of g3, depth-first: Pos, Vel, Health, Tag.
    try t.expectEqual(@as(usize, 4), T.countQueryLeaves(g3));
    try t.expect(T.queryLeafAt(g3, 0) == Pos);
    try t.expect(T.queryLeafAt(g3, 1) == Vel);
    try t.expect(T.queryLeafAt(g3, 2) == Health);
    try t.expect(T.queryLeafAt(g3, 3) == Tag);

    // Empty groups vanish at any level.
    try t.expectEqual(@as(usize, 0), T.countQueryLeaves(.{ .{}, .{} }));
    try t.expectEqual(@as(usize, 2), T.countQueryLeaves(.{ g1, .{} }));
    try t.expectEqual(@as(usize, 0), T.countUniqueQueryLeaves(.{ .{}, .{} }));

    // Four levels of nesting.
    const g4 = .{ .{.{.{Pos}}}, Vel };
    try t.expectEqual(@as(usize, 2), T.countQueryLeaves(g4));
    try t.expect(T.queryLeafAt(g4, 0) == Pos);
    try t.expect(T.queryLeafAt(g4, 1) == Vel);

    // Dedup keeps the first occurrence: Pos,Pos,Vel,Tag,Vel -> Pos,Vel,Tag.
    const d = .{ Pos, g1, Tag, Vel };
    try t.expectEqual(@as(usize, 5), T.countQueryLeaves(d));
    try t.expectEqual(@as(usize, 3), T.countUniqueQueryLeaves(d));
    try t.expect(T.isFirstQueryOccurrence(d, 0));
    try t.expect(!T.isFirstQueryOccurrence(d, 1));
    try t.expect(T.isFirstQueryOccurrence(d, 2));
    try t.expect(T.isFirstQueryOccurrence(d, 3));
    try t.expect(!T.isFirstQueryOccurrence(d, 4));

    // Dedup through nesting: .{g1, Pos, .{Vel}} -> Pos, Vel.
    const dn = .{ g1, Pos, .{Vel} };
    try t.expectEqual(@as(usize, 2), T.countUniqueQueryLeaves(dn));

    // Containment sees through groups.
    try t.expect(T.queryContains(g3, Vel));
    try t.expect(T.queryContains(g3, Tag));
    try t.expect(!T.queryContains(g3, Ghost));
    try t.expect(!T.queryContains(.{ .{}, .{ g1, .{} } }, Ghost));
    try t.expect(T.queryContains(.{.{.{Ghost}}}, Ghost));
}

test "nested value helpers: flat leaf types over 3+ levels" {
    const T = ECSTable(.t302);

    const VT = @TypeOf(.{
        Pos{ .x = 1, .y = 2 },
        .{ Vel{ .dx = 3, .dy = 4 }, Tag{} },
        .{.{Health{ .hp = 5 }}},
        .{},
    });
    // Depth-first: Pos, Vel, Tag, Health.
    try t.expectEqual(@as(usize, 4), T.countValueLeaves(VT));
    try t.expect(T.valueLeafTypeAt(VT, 0) == Pos);
    try t.expect(T.valueLeafTypeAt(VT, 1) == Vel);
    try t.expect(T.valueLeafTypeAt(VT, 2) == Tag);
    try t.expect(T.valueLeafTypeAt(VT, 3) == Health);

    // Empty groups contribute zero leaves at any depth.
    try t.expectEqual(@as(usize, 0), T.countValueLeaves(@TypeOf(.{ .{}, .{.{}} })));
    try t.expectEqual(@as(usize, 1), T.countValueLeaves(@TypeOf(.{ .{}, Tag{} })));

    // Four levels, single leaf.
    const Deep = @TypeOf(.{.{.{.{Pos{ .x = 0, .y = 0 }}}}});
    try t.expectEqual(@as(usize, 1), T.countValueLeaves(Deep));
    try t.expect(T.valueLeafTypeAt(Deep, 0) == Pos);
}

test "ECSTable nested create: groups expand depth-first with payloads" {
    const T = ECSTable(.t303);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    _ = try T.addComponent(t.allocator, Vel);
    _ = try T.addComponent(t.allocator, Health);
    _ = try T.addComponent(t.allocator, Tag);

    // Reusable group mixing data and flag instances, plus empty groups.
    const base = .{ Pos{ .x = 1, .y = 2 }, Tag{} };
    const mid = .{ base, Vel{ .dx = 3, .dy = 4 }, .{} };
    const e = try T.create(t.allocator, .{ mid, Health{ .hp = 9 }, .{.{}} });
    try t.expect(try e.isComponentActive(Pos));
    try t.expect(try e.isComponentActive(Vel));
    try t.expect(try e.isComponentActive(Health));
    try t.expect(try e.isComponentActive(Tag));
    try t.expectEqual(Pos{ .x = 1, .y = 2 }, try e.getComponent(Pos));
    try t.expectEqual(Vel{ .dx = 3, .dy = 4 }, try e.getComponent(Vel));
    try t.expectEqual(Health{ .hp = 9 }, try e.getComponent(Health));

    // Runtime values inside groups land in the column verbatim.
    const got = try e.getComponent(Pos);
    const rg = .{got};
    const f = try T.create(t.allocator, .{ .{rg}, Tag{} });
    try t.expectEqual(Pos{ .x = 1, .y = 2 }, try f.getComponent(Pos));
    try t.expect(try f.isComponentActive(Tag));
    try t.expect(!try f.isComponentActive(Vel));

    // Duplicate flags across groups dedup silently (idempotent setBit).
    const g = try T.create(t.allocator, .{ Tag{}, .{ Tag{}, .{} } });
    try t.expect(try g.isComponentActive(Tag));

    // createN reuses the same nested group per entity.
    const Ctx = struct {
        count: u32 = 0,
        last: T.EntityReference = .{ .slot = 0, .gen = 0 },

        fn push(self: *@This(), ref: T.EntityReference) void {
            self.count += 1;
            self.last = ref;
        }
    };
    var ctx = Ctx{};
    try T.createN(t.allocator, .{mid}, 3, &ctx, Ctx.push);
    try t.expectEqual(@as(u32, 3), ctx.count);
    try t.expect(ctx.last.isValid());
    try t.expectEqual(Pos{ .x = 1, .y = 2 }, try ctx.last.getComponent(Pos));
    try t.expect(try ctx.last.isComponentActive(Tag));

    // Missing registration is still reported through nesting.
    try t.expectError(T.Error.ComponentNotFound, T.create(t.allocator, .{.{Ghost{ .v = 1 }}}));
    try t.expectError(T.Error.ComponentNotFound, T.create(t.allocator, .{ Pos{ .x = 0, .y = 0 }, .{.{Ghost{ .v = 2 }}} }));
}

test "ECSTable nested Query: groups, dedup, excludes at depth" {
    const T = ECSTable(.t304);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    _ = try T.addComponent(t.allocator, Vel);
    _ = try T.addComponent(t.allocator, Tag);
    _ = try T.addComponent(t.allocator, Health);

    // Rows: 0:Pos 1:Pos+Vel 2:Pos+Vel+Tag 3:Vel+Tag 4:Pos+Tag+Health.
    _ = try T.create(t.allocator, .{Pos{ .x = 0, .y = 0 }});
    _ = try T.create(t.allocator, .{ Pos{ .x = 1, .y = 1 }, Vel{ .dx = 1, .dy = 1 } });
    _ = try T.create(t.allocator, .{ Pos{ .x = 2, .y = 2 }, .{ Vel{ .dx = 2, .dy = 2 }, Tag{} } });
    _ = try T.create(t.allocator, .{ Vel{ .dx = 3, .dy = 3 }, Tag{} });
    _ = try T.create(t.allocator, .{ .{ Pos{ .x = 4, .y = 4 }, Tag{} }, Health{ .hp = 4 } });

    const g = .{ Pos, Vel };

    // Group in Includes behaves like the flat list.
    const Q1 = T.Query(.{ g, Tag }, .{}, .forward, *Collect, Collect.push);
    var c = Collect{};
    try t.expect(try Q1.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{2}, c.rows[0..c.n]);

    // Duplicates across nesting levels dedup to the same match.
    const Q2 = T.Query(.{ Pos, .{ Pos, Vel }, Tag, .{ Vel, Tag } }, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q2.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{2}, c.rows[0..c.n]);

    // Nested group in Excludes.
    const Q3 = T.Query(.{Pos}, .{.{Vel}}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q3.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 4 }, c.rows[0..c.n]);

    // Deep nesting plus empty groups on both sides.
    const Q4 = T.Query(.{ .{.{Pos}}, .{} }, .{.{.{Health}}}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q4.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, c.rows[0..c.n]);

    // Nested group passed whole, without an outer wrapper element.
    const Q5 = T.Query(g, .{.{Health}}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q5.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 1, 2 }, c.rows[0..c.n]);
}

test "bare single struct without tuple: create/createN/Query" {
    const T = ECSTable(.t305);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    _ = try T.addComponent(t.allocator, Vel);
    _ = try T.addComponent(t.allocator, Tag);

    // Bare data instance behaves like a 1-tuple.
    const a = try T.create(t.allocator, Pos{ .x = 1, .y = 2 });
    try t.expect(try a.isComponentActive(Pos));
    try t.expect(!try a.isComponentActive(Vel));
    try t.expectEqual(Pos{ .x = 1, .y = 2 }, try a.getComponent(Pos));

    // Bare flag instance only sets the bit.
    const b = try T.create(t.allocator, Tag{});
    try t.expect(try b.isComponentActive(Tag));
    try t.expect(!try b.isComponentActive(Pos));

    // Bare unregistered component still reports ComponentNotFound.
    try t.expectError(T.Error.ComponentNotFound, T.create(t.allocator, Ghost{ .v = 1 }));

    // Bare value in createN.
    const Ctx = struct {
        count: u32 = 0,
        last: T.EntityReference = .{ .slot = 0, .gen = 0 },

        fn push(self: *@This(), ref: T.EntityReference) void {
            self.count += 1;
            self.last = ref;
        }
    };
    var ctx = Ctx{};
    try T.createN(t.allocator, Vel{ .dx = 5, .dy = 6 }, 2, &ctx, Ctx.push);
    try t.expectEqual(@as(u32, 2), ctx.count);
    try t.expectEqual(Vel{ .dx = 5, .dy = 6 }, try ctx.last.getComponent(Vel));

    // Rows so far: 0:Pos 1:Tag 2:Vel 3:Vel.
    // Bare Includes behaves like .{Pos}.
    const Q1 = T.Query(Pos, .{}, .forward, *Collect, Collect.push);
    var c = Collect{};
    try t.expect(try Q1.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{0}, c.rows[0..c.n]);

    // Bare Excludes behaves like .{Vel}.
    const Q2 = T.Query(.{}, Vel, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q2.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 1 }, c.rows[0..c.n]);

    // Bare on both sides.
    const Q3 = T.Query(Pos, Vel, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q3.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{0}, c.rows[0..c.n]);

    // Bare mixed with a group.
    const Q4 = T.Query(.{ Tag, Pos }, Vel, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q4.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{}, c.rows[0..c.n]);

    // Bare flag query matches flag rows.
    const Q5 = T.Query(Tag, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q5.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{1}, c.rows[0..c.n]);
}

test "ECSTable tag isolates instances" {
    const A = ECSTable(.tag_a);
    const B = ECSTable(.tag_b);
    const A2 = ECSTable(.tag_a);
    try A.init();
    defer A.deinit(t.allocator);
    try B.init();
    defer B.deinit(t.allocator);
    // A and A2 share storage; B is fully isolated.

    try t.expect(A.tag_value == .tag_a);
    try t.expect(B.tag_value == .tag_b);
    try t.expect(A.EntityReference != B.EntityReference);

    const ea: A.EntityReference = try A.create(t.allocator, .{});
    try t.expectEqual(@as(u32, 1), A.rowCount());
    try t.expectEqual(@as(u32, 0), B.rowCount());
    try t.expectEqual(@as(u32, 1), A2.rowCount());

    // Same tag sees the same row through a re-resolved handle…
    const rea = try A2.rowToEntity(0);
    try t.expectEqual(ea.slot, rea.slot);
    try t.expectEqual(ea.gen, rea.gen);
    // …while the other tag stays empty.
    try t.expectError(B.Error.RowOutOfBounds, B.rowToEntity(0));

    const eb: B.EntityReference = try B.create(t.allocator, .{});
    try t.expectEqual(@as(u32, 1), A.rowCount());
    try t.expectEqual(@as(u32, 1), B.rowCount());
    try t.expect(ea.isValid());
    try t.expect(eb.isValid());

    try ea.destroy();
    try t.expect(!ea.isValid());
    try t.expect(eb.isValid());
}

test "ECSTable enum flags: add/remove/contain parallel API" {
    const T = ECSTable(.t401);
    try T.init();
    defer T.deinit(t.allocator);

    try t.expect(!T.containFlag(.flying));
    try t.expect(try T.addFlag(t.allocator, .flying));
    try t.expect(T.containFlag(.flying));
    try t.expect(!T.containFlag(.swimming));
    // Idempotent re-add is a no-op.
    try t.expect(!try T.addFlag(t.allocator, .flying));
    try t.expect(try T.addFlag(t.allocator, .swimming));
    try t.expect(T.containFlag(.swimming));

    // Struct flags and enum flags share flag_cols but never collide:
    // Tag has @typeName key, .flying has "@EnumLiteral(.flying)" key.
    try t.expect(try T.addComponent(t.allocator, Tag));
    try t.expect(T.containComponent(Tag));
    try t.expect(T.containFlag(.flying));

    try t.expect(try T.removeFlag(t.allocator, .flying));
    try t.expect(!T.containFlag(.flying));
    try t.expect(T.containFlag(.swimming));
    try t.expect(!try T.removeFlag(t.allocator, .flying));
    // Struct flag untouched by enum removal.
    try t.expect(T.containComponent(Tag));
}

test "ECSTable enum flags: create/set/activity mix with structs" {
    const T = ECSTable(.t402);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    _ = try T.addComponent(t.allocator, Tag);
    _ = try T.addFlag(t.allocator, .flying);
    _ = try T.addFlag(t.allocator, .swimming);

    // Mixed data + struct flag + enum flag in one tuple.
    const e = try T.create(t.allocator, .{ Pos{ .x = 1, .y = 2 }, Tag{}, .flying });
    try t.expect(try e.isComponentActive(Pos));
    try t.expect(try e.isComponentActive(Tag));
    try t.expect(try e.isFlagActive(.flying));
    try t.expect(!try e.isFlagActive(.swimming));
    try t.expect(!try T.Row.isFlagActive(0, .swimming));

    // Bare enum flag behaves like bare Tag{}.
    const f = try T.create(t.allocator, .flying);
    try t.expect(try f.isFlagActive(.flying));
    try t.expect(!try f.isComponentActive(Pos));

    // setComponent with enum only sets the bit.
    const g = try T.create(t.allocator, .{});
    try t.expect(!try g.isFlagActive(.swimming));
    try g.setComponent(.swimming);
    try t.expect(try g.isFlagActive(.swimming));
    try T.Row.setComponent(2, .flying);
    try t.expect(try T.Row.isFlagActive(2, .flying));

    // setFlag / setFlagActivity sugar (Row + EntityReference).
    try T.Row.setFlagActivity(0, .swimming, .active);
    try t.expect(try T.Row.isFlagActive(0, .swimming));
    try e.setFlag(.swimming);
    try t.expect(try e.isFlagActive(.swimming));
    try e.setFlagActivity(.swimming, .inactive);
    try t.expect(!try e.isFlagActive(.swimming));

    // Duplicate enum flags dedup silently, like struct flags.
    const h = try T.create(t.allocator, .{ .flying, .{.flying} });
    try t.expect(try h.isFlagActive(.flying));

    // Nested groups mix structs and enums depth-first.
    const base = .{ Pos{ .x = 3, .y = 4 }, .flying };
    const i = try T.create(t.allocator, .{ base, Tag{}, .{ .swimming, .{} } });
    try t.expect(try i.isComponentActive(Pos));
    try t.expect(try i.isComponentActive(Tag));
    try t.expect(try i.isFlagActive(.flying));
    try t.expect(try i.isFlagActive(.swimming));

    // Unregistered enum reports ComponentNotFound everywhere.
    try t.expectError(T.Error.ComponentNotFound, T.create(t.allocator, .{.ghost}));
    try t.expectError(T.Error.ComponentNotFound, T.create(t.allocator, .{ Pos{ .x = 0, .y = 0 }, .{.ghost} }));
    try t.expectError(T.Error.ComponentNotFound, e.setFlagActivity(.ghost, .active));
    try t.expectError(T.Error.ComponentNotFound, T.Row.isFlagActive(0, .ghost));

    // createN with bare enum flag.
    const Ctx = struct {
        count: u32 = 0,
        last: T.EntityReference = .{ .slot = 0, .gen = 0 },
        fn push(self: *@This(), ref: T.EntityReference) void {
            self.count += 1;
            self.last = ref;
        }
    };
    var ctx = Ctx{};
    try T.createN(t.allocator, .swimming, 2, &ctx, Ctx.push);
    try t.expectEqual(@as(u32, 2), ctx.count);
    try t.expect(try ctx.last.isFlagActive(.swimming));
}

test "ECSTable enum flags: Query mixed, nested, dedup, excludes" {
    const T = ECSTable(.t403);
    try T.init();
    defer T.deinit(t.allocator);

    _ = try T.addComponent(t.allocator, Pos);
    _ = try T.addComponent(t.allocator, Vel);
    _ = try T.addComponent(t.allocator, Tag);
    _ = try T.addFlag(t.allocator, .flying);
    _ = try T.addFlag(t.allocator, .swimming);

    // Rows: 0:Pos+flying 1:Pos+Vel+flying+swimming 2:Vel+swimming 3:Pos+Tag.
    _ = try T.create(t.allocator, .{ Pos{ .x = 0, .y = 0 }, .flying });
    _ = try T.create(t.allocator, .{ Pos{ .x = 1, .y = 1 }, Vel{ .dx = 1, .dy = 1 }, .flying, .swimming });
    _ = try T.create(t.allocator, .{ Vel{ .dx = 2, .dy = 2 }, .swimming });
    _ = try T.create(t.allocator, .{ Pos{ .x = 3, .y = 3 }, Tag{} });

    // Helpers see through nesting; enum identity is tag name, not type.
    const g = .{ Pos, .flying };
    try t.expectEqual(@as(usize, 2), T.countQueryLeaves(g));
    try t.expect(T.queryContains(g, Pos));
    try t.expect(!T.queryContains(g, Vel));
    try t.expect(T.queryContainsFlag(g, .flying));
    try t.expect(!T.queryContainsFlag(g, .swimming));
    try t.expectEqual(@as(usize, 2), T.countUniqueQueryLeaves(.{ Pos, .{ Pos, .flying }, .flying }));
    try t.expect(T.queryContainsFlag(.{.{.flying}}, .flying));

    // Includes enum flag.
    const Q1 = T.Query(.{.flying}, .{}, .forward, *Collect, Collect.push);
    var c = Collect{};
    try t.expect(try Q1.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 1 }, c.rows[0..c.n]);

    // Mixed struct + enum includes.
    const Q2 = T.Query(.{ Pos, .flying }, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q2.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 1 }, c.rows[0..c.n]);

    // Enum in excludes: Pos without swimming.
    const Q3 = T.Query(.{Pos}, .{.swimming}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q3.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 3 }, c.rows[0..c.n]);

    // Bare enum on both sides.
    const Q4 = T.Query(.flying, .swimming, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q4.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{0}, c.rows[0..c.n]);

    // Nested group with enum + dedup across levels.
    const Q5 = T.Query(.{ g, .swimming }, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q5.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{1}, c.rows[0..c.n]);

    const Q6 = T.Query(.{ Pos, .{ Pos, .flying }, .flying }, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expect(try Q6.iterateAll(&c, null, null, null));
    try t.expectEqualSlices(u32, &[_]u32{ 0, 1 }, c.rows[0..c.n]);

    // Unregistered enum in query reports ComponentNotFound at iterateAll.
    const QGhost = T.Query(.{.ghost}, .{}, .forward, *Collect, Collect.push);
    c = Collect{};
    try t.expectError(T.Error.ComponentNotFound, QGhost.iterateAll(&c, null, null, null));
}
