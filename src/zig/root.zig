const std = @import("std");
const cuda = @import("wmma");
const testing = std.testing;
const assert = std.debug.assert;

pub fn wmmBlock(a: []f16, b: []f16, c: []f16, stride: usize, allocator: std.mem.Allocator) ![]f16 {
    return (try cuda.wmmBlock(a, b, c, stride, allocator)).items;
}

pub fn MMA(a: []f16, b: []f16, side_size: usize, alloc: std.mem.Allocator) ![]f16 {
    assert(side_size % 16 == 0);
    assert(a.len == side_size * side_size);
    assert(b.len == side_size * side_size);

    var c = try alloc.alloc(f16, side_size * side_size);
    const n_blocks = side_size * side_size / 256;
    const side_blocks = side_size / 16;
    for (0..n_blocks) |i| {
        for (0..n_blocks) |j| {
            const c_new = try wmmBlock(a[(i / side_blocks * side_size * 16 + i % side_blocks * 16)..], b[(j / side_blocks * side_size * 16 + j % side_blocks * 16)..], c[(i / side_blocks * side_size * 16 + i % side_blocks * 16)..], side_size / 16 - 1, alloc);
            alloc.free(c);
            c = c_new;
        }
    }
    return c;
}

test "single block multiply" {
    const alloc = std.testing.allocator;
    var cok = [_]f16{1} ** 256;
    const e = try MMA(&cok, &cok, 16, alloc);
    defer alloc.free(e);
    std.debug.print("{d:.3}/n", .{e});
}
