const std = @import("std");

const ThisBuild = @This();

pub const Options = struct {
    target: ?std.Build.ResolvedTarget = null,
    optimize: ?std.builtin.OptimizeMode = null,
    dependency_bit_tree: ?*std.Build.Module = null,

    pub fn initFromOptions(b: *std.Build) Options {
        return .{
            .target = b.standardTargetOptions(.{}),
            .optimize = b.standardOptimizeOption(.{}),
        };
    }

    pub fn getModule(self: Options, b: *std.Build) *std.Build.Module {
        const target = self.target orelse b.standardTargetOptions(.{});
        const optimize = self.optimize orelse b.standardOptimizeOption(.{});
        const self_dep = b.dependencyFromBuildZig(ThisBuild, .{
            .target = target,
            .optimize = optimize,
        });
        const bit_tree_mod = self.dependency_bit_tree orelse self_dep.builder.dependency("bit_tree", .{
            .target = target,
            .optimize = optimize,
        }).module("bit_tree");
        const mod = b.createModule(.{
            .root_source_file = self_dep.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        });
        mod.addImport("bit_tree", bit_tree_mod);
        return mod;
    }
};

fn createModuleOwn(b: *std.Build, options: Options) *std.Build.Module {
    const target = options.target orelse b.standardTargetOptions(.{});
    const optimize = options.optimize orelse b.standardOptimizeOption(.{});
    const bit_tree_mod = options.dependency_bit_tree orelse b.dependency("bit_tree", .{
        .target = target,
        .optimize = optimize,
    }).module("bit_tree");
    const mod = b.addModule("t_ecs", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("bit_tree", bit_tree_mod);
    return mod;
}

pub fn build(b: *std.Build) void {
    const options = Options.initFromOptions(b);
    const target = options.target orelse b.standardTargetOptions(.{});
    const optimize = options.optimize orelse b.standardOptimizeOption(.{});

    const mod = createModuleOwn(b, options);

    const bit_tree_mod = mod.import_table.get("bit_tree").?;

    const exe = b.addExecutable(.{
        .name = "t_ecs",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "t_ecs", .module = mod },
                .{ .name = "bit_tree", .module = bit_tree_mod },
            },
        }),
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
