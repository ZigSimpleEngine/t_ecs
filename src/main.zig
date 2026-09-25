const std = @import("std");

const t_ecs = @import("t_ecs");

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const T = t_ecs.ECSTable(0);
    try T.init();
    defer T.deinit(alloc);

    const Ctx = struct {
        fn push(_: void, ref: T.EntityReference) void {
            std.debug.print("created slot={d} gen={d}\n", .{ ref.slot, ref.gen });
        }
    };
    try T.createN(alloc, .{}, 3, {}, Ctx.push);
    const first = try T.rowToEntity(0);
    try first.destroy();
    std.debug.print("rows={d} slots={d} first_valid={}\n", .{ T.rowCount(), T.slotCount(), first.isValid() });

    _ = init;
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa);
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
