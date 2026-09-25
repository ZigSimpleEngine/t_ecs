const std = @import("std");
const bit_tree = @import("bit_tree");

const Tree = bit_tree.BitTree(.u64);
const BitState = bit_tree.BitState;
const Allocator = std.mem.Allocator;

/// Fixed-identity table id tag is the `table_id` comptime parameter itself:
/// every `EntityReference` belongs to its `ECSTable(id)` instantiation by type.
pub fn ECSTable(comptime table_id: u32) type {
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
        pub const table: u32 = table_id;

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
        const FlagColumn = struct {
            /// Owned `@typeName(T)` copy, map key for the column.
            name: []u8,
            /// Per-row activity flags.
            tree: Tree = .{},
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
            pub fn getComponent(self: EntityReference, comptime T: type) Error!T {
                return Row.getComponent(try Table.resolveRef(self), T);
            }

            /// Returns a mutable pointer into the entity row payload.
            pub fn getComponentPtr(self: EntityReference, comptime T: type) Error!*T {
                return Row.getComponentPtr(try Table.resolveRef(self), T);
            }

            /// Overwrites the component payload of the entity row.
            /// The value type selects the column, no separate type tag.
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
            pub fn getComponent(row: u32, comptime T: type) Error!T {
                if (comptime !hasFields(T)) @compileError("flag components carry no payload");
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
            pub fn getComponentPtr(row: u32, comptime T: type) Error!*T {
                if (comptime !hasFields(T)) @compileError("flag components carry no payload");
                if (!initialized) return Error.NotInitialized;
                if (row >= destroyed.bitset.bits_count) return Error.RowOutOfBounds;
                if (destroyed.bitset.getBit(row) == .active) return Error.AlreadyDestroyed;
                const col = try Table.dataColumn(T);
                return @ptrCast(@alignCast(Table.rowBytes(col, row).ptr));
            }

            /// Overwrites the component payload of a live row.
            /// The value type selects the column, no separate type tag.
            pub fn setComponent(row: u32, component: anytype) Error!void {
                const T = @TypeOf(component);
                if (comptime !hasFields(T)) @compileError("flag components carry no payload");
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
        };

        /// Resolves the activity tree of a registered component,
        /// choosing the payload or flag column at comptime.
        fn componentTree(comptime T: type) Error!*Tree {
            if (hasFields(T)) {
                const col = try dataColumn(T);
                return &col.tree;
            } else {
                const idx = flag_map.get(@typeName(T)) orelse return Error.ComponentNotFound;
                return &flag_cols.items[idx].tree;
            }
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
            return ti.@"struct".fields.len > 0;
        }

        /// Registers a component type. Creates one column sized to the
        /// current row count, all flags inactive, payload bytes zeroed.
        /// Duplicate registration is an error, not a silent no-op.
        pub fn addComponent(alloc: Allocator, comptime T: type) Error!void {
            if (!initialized) return Error.NotInitialized;
            if (hasFields(T)) {
                if (data_map.contains(@typeName(T))) return Error.ComponentAlreadyExists;
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
                if (flag_map.contains(@typeName(T))) return Error.ComponentAlreadyExists;
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
        }

        /// Unregisters a component type. Removes its column with swap-remove
        /// and repoints the map entry of the moved column by its stored name.
        pub fn removeComponent(alloc: Allocator, comptime T: type) Error!void {
            if (!initialized) return Error.NotInitialized;
            if (hasFields(T)) {
                const idx = data_map.get(@typeName(T)) orelse return Error.ComponentNotFound;
                _ = data_map.remove(@typeName(T));
                var gone = data_cols.swapRemove(idx);
                if (idx < data_cols.items.len) {
                    try data_map.put(alloc, data_cols.items[idx].name, idx);
                }
                freeDataColumn(alloc, &gone);
            } else {
                const idx = flag_map.get(@typeName(T)) orelse return Error.ComponentNotFound;
                _ = flag_map.remove(@typeName(T));
                var gone = flag_cols.swapRemove(idx);
                if (idx < flag_cols.items.len) {
                    try flag_map.put(alloc, flag_cols.items[idx].name, idx);
                }
                gone.tree.deinit(alloc);
                alloc.free(gone.name);
            }
        }

        /// Checks registration in the matching map, chosen at comptime.
        pub fn containComponent(comptime T: type) bool {
            if (!initialized) return false;
            if (hasFields(T)) return data_map.contains(@typeName(T));
            return flag_map.contains(@typeName(T));
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
        /// An empty tuple creates a bare entity. Duplicate types and flag
        /// components in the tuple are comptime errors. Every listed type
        /// must be registered; listed columns start active with copied
        /// payloads, every other column starts inactive on the new row.
        pub fn create(alloc: Allocator, comptime values: anytype) Error!EntityReference {
            if (!initialized) return Error.NotInitialized;
            validateTuple(values);
            inline for (values) |v| {
                if (!containComponent(@TypeOf(v))) return Error.ComponentNotFound;
            }
            const row = try allocRow(alloc);
            for (data_cols.items) |*col| col.tree.setBit(row, .inactive);
            for (flag_cols.items) |*col| col.tree.setBit(row, .inactive);
            inline for (values) |v| {
                const col = try dataColumn(@TypeOf(v));
                col.tree.setBit(row, .active);
                @memcpy(rowBytes(col, row), std.mem.asBytes(&v));
            }
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
        pub fn createN(alloc: Allocator, comptime values: anytype, n: u32, context: anytype, comptime cb: fn (@TypeOf(context), EntityReference) void) Error!void {
            if (!initialized) return Error.NotInitialized;
            var i: u32 = 0;
            while (i < n) : (i += 1) {
                cb(context, try create(alloc, values));
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
                _ = It.iterateAll(.{ .tree = &destroyed, .context = &ctx }, null, null);
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
                _ = It.iterateAll(.{ .tree = &destroyed, .context = &ctx }, live, null);
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
            comptime on_row: fn (ctx: Context, row: u32) callconv(.@"inline") bool,
        ) type {
            comptime validateQuery(Includes, Excludes);
            const IL = Includes.len;
            const EL = Excludes.len;
            return struct {
                /// Runs the match over an optional bit range with an optional
                /// entity activity filter. Returns false on early callback exit.
                pub fn iterateAll(ctx: Context, start_bit: ?u32, end_bit: ?u32, entity_activity: ?bool) Error!bool {
                    if (!initialized) return Error.NotInitialized;
                    var inc: [IL]*Tree = undefined;
                    inline for (Includes, 0..) |C, k| inc[k] = try Table.componentTree(C);
                    var exc: [EL + 1]*Tree = undefined;
                    inline for (Excludes, 0..) |C, k| exc[k] = try Table.componentTree(C);
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

        /// Comptime shape check of a query: both sides are tuples of struct
        /// types with no duplicates inside or across the tuple pair.
        fn validateQuery(Includes: anytype, Excludes: anytype) void {
            const iti = @typeInfo(@TypeOf(Includes));
            const eti = @typeInfo(@TypeOf(Excludes));
            if (iti != .@"struct" or !iti.@"struct".is_tuple) @compileError("Includes must be a tuple of component types");
            if (eti != .@"struct" or !eti.@"struct".is_tuple) @compileError("Excludes must be a tuple of component types");
            inline for (0..iti.@"struct".fields.len) |k| checkQueryType(Includes[k]);
            inline for (0..eti.@"struct".fields.len) |k| checkQueryType(Excludes[k]);
            inline for (0..iti.@"struct".fields.len) |i| {
                inline for (0..i) |j| {
                    if (Includes[j] == Includes[i]) @compileError("duplicate component type in Includes");
                }
                inline for (0..eti.@"struct".fields.len) |j| {
                    if (Excludes[j] == Includes[i]) @compileError("component type in both Includes and Excludes");
                }
            }
        }

        /// Query tuple elements must be struct types, values are rejected.
        fn checkQueryType(X: anytype) void {
            if (@TypeOf(X) != type) @compileError("query tuples hold component types, not values");
            if (@typeInfo(X) != .@"struct") @compileError("component must be a struct type");
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

        /// Comptime shape check of a create tuple: tuple-ness, struct
        /// payload values only, no duplicate types.
        fn validateTuple(comptime values: anytype) void {
            const ti = @typeInfo(@TypeOf(values));
            if (ti != .@"struct" or !ti.@"struct".is_tuple) @compileError("create expects a tuple of component values");
            inline for (ti.@"struct".fields, 0..) |f, k| {
                const fti = @typeInfo(f.type);
                if (fti != .@"struct") @compileError("tuple element must be a component struct value");
                if (fti.@"struct".fields.len == 0) @compileError("flag components carry no payload, omit them from create");
                inline for (ti.@"struct".fields[0..k]) |g| {
                    if (g.type == f.type) @compileError("duplicate component type in create tuple");
                }
            }
        }

        /// Collects destroyed row ids in forward order.
        const VictimCtx = struct {
            list: *std.ArrayListUnmanaged(u32),
            alloc: Allocator,
            oom: bool = false,

            inline fn push(ctx: *VictimCtx, id: u32) bool {
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

            inline fn push(ctx: *DonorCtx, id: u32) bool {
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
            _ = It.iterateAll(.{ .tree = &destroyed, .context = &finder }, null, null);
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

            inline fn push(ctx: *ReuseCtx, id: u32) bool {
                ctx.row = id;
                return false;
            }
        };
    };
}

const t = std.testing;

test "ECSTable core: create/isValid/destroy lifecycle" {
    const T = ECSTable(101);
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
    const T = ECSTable(102);
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
    const T = ECSTable(103);
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
    const T = ECSTable(104);
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
    const T = ECSTable(105);
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
    const T = ECSTable(201);
    try T.init();
    defer T.deinit(t.allocator);

    try t.expect(!T.containComponent(Pos));
    try T.addComponent(t.allocator, Pos);
    try T.addComponent(t.allocator, Vel);
    try T.addComponent(t.allocator, Tag);
    try t.expect(T.containComponent(Pos));
    try t.expect(T.containComponent(Vel));
    try t.expect(T.containComponent(Tag));
    try t.expect(!T.containComponent(Health));
    try t.expectError(T.Error.ComponentAlreadyExists, T.addComponent(t.allocator, Pos));
    try t.expectError(T.Error.ComponentAlreadyExists, T.addComponent(t.allocator, Tag));

    const e = try T.create(t.allocator, .{ Pos{ .x = 1, .y = 2 }, Vel{ .dx = 3, .dy = 4 } });
    try t.expectEqual(Pos{ .x = 1, .y = 2 }, try e.getComponent(Pos));

    try T.removeComponent(t.allocator, Pos);
    try t.expect(!T.containComponent(Pos));
    try t.expect(T.containComponent(Vel));
    try t.expectError(T.Error.ComponentNotFound, T.removeComponent(t.allocator, Pos));
    try t.expectError(T.Error.ComponentNotFound, e.getComponent(Pos));
    try t.expectEqual(Vel{ .dx = 3, .dy = 4 }, try e.getComponent(Vel));

    try T.removeComponent(t.allocator, Tag);
    try t.expect(!T.containComponent(Tag));
    try T.addComponent(t.allocator, Pos);
    try t.expect(T.containComponent(Pos));
}

test "ECSTable columns: late add sizes rows, get/set/ptr roundtrip" {
    const T = ECSTable(202);
    try T.init();
    defer T.deinit(t.allocator);

    const a = try T.create(t.allocator, .{});
    const b = try T.create(t.allocator, .{});
    const c = try T.create(t.allocator, .{});
    try T.addComponent(t.allocator, Health);
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
    const T = ECSTable(203);
    try T.init();
    defer T.deinit(t.allocator);

    try T.addComponent(t.allocator, Pos);
    try T.addComponent(t.allocator, Vel);
    try T.addComponent(t.allocator, Health);
    try T.addComponent(t.allocator, Tag);

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
    const T = ECSTable(204);
    try T.init();
    defer T.deinit(t.allocator);

    try T.addComponent(t.allocator, Pos);
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

    inline fn push(self: *Collect, row: u32) bool {
        self.rows[self.n] = row;
        self.n += 1;
        return true;
    }
};

test "ECSTable Query: truth table, idioms, direction, range, filters" {
    const T = ECSTable(205);
    try T.init();
    defer T.deinit(t.allocator);

    try T.addComponent(t.allocator, Pos);
    try T.addComponent(t.allocator, Vel);
    try T.addComponent(t.allocator, Tag);

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
