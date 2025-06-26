const std = @import("std");
const cuda = @import("wmma");
const testing = std.testing;
const assert = std.debug.assert;
const print = std.debug.print;

pub fn wmmBlock(a: []f16, b: []f16, c: []f16, stride: usize, allocator: std.mem.Allocator) ![]f16 {
    return (try cuda.wmmBlock(a, b, c, stride, allocator)).items;
}

pub fn MMA(a: []f16, b: []f16, side_size: u32, alloc: std.mem.Allocator) ![]f16 {
    assert(side_size % 16 == 0);
    assert(a.len == side_size * side_size);
    assert(b.len == side_size * side_size);

    var c = try alloc.alloc(f16, side_size * side_size);
    for (0..c.len) |i|
        c[i] = 0;
    const n_blocks = side_size * side_size / 256;
    const side_blocks = side_size / 16;
    for (0..n_blocks) |n| {
        const a_line = n % side_blocks;
        const b_line = n / side_blocks;
        for (0..side_blocks) |i| {
            const a_offset = a_line * 16 + i * side_size * 16;
            const b_offset = b_line * 16 + i * side_size * 16;
            const c_offset = a_line * 16 + b_line * side_size * 16;
            const c_new = try wmmBlock(a[a_offset..], b[b_offset..], c[c_offset..], side_size, alloc);
            defer alloc.free(c_new);
            for (0..16) |x| {
                for (0..16) |y|
                    c[c_offset + x + y * side_size] += c_new[x + y * side_size];
            }
        }
    }
    return c;
}

test "single block ones multiply" {
    const alloc = std.testing.allocator;
    var mat = [_]f16{1} ** 256;
    const e = try MMA(&mat, &mat, 16, alloc);
    defer alloc.free(e);
    for (e) |r| assert(r == 16.0);
}

test "4 block ones multiply" {
    const alloc = std.testing.allocator;
    var mat = [_]f16{1} ** (256 * 4);
    const e = try MMA(&mat, &mat, 32, alloc);
    defer alloc.free(e);
    for (e) |r| assert(r == 32.0);
}

test "9 block ones multiply" {
    const alloc = std.testing.allocator;
    var mat = [_]f16{1} ** (256 * 9);
    const e = try MMA(&mat, &mat, 48, alloc);
    defer alloc.free(e);
    for (e) |r| assert(r == 48.0);
}

test "actual multiplication" {
    const alloc = std.testing.allocator;
    var a = [_]f16{0} ** 256;
    a[0] = 1;
    a[8] = 2;
    a[128] = 3;
    a[136] = 4;
    var b = [_]f16{0} ** 256;
    b[0] = 5;
    b[8] = 6;
    b[128] = 7;
    b[136] = 8;
    const e = try MMA(&a, &b, 16, alloc);
    defer alloc.free(e);
    assert(e[0] == 26);
    assert(e[8] == 38);
    assert(e[128] == 30);
    assert(e[136] == 44);
}

test "actual multiplication 2" {
    const alloc = std.testing.allocator;
    var a = [_]f16{0} ** (256 * 4);
    a[0] = 1;
    a[16] = 2;
    a[512] = 3;
    a[528] = 4;
    var b = [_]f16{0} ** (256 * 4);
    b[0] = 5;
    b[16] = 6;
    b[512] = 7;
    b[528] = 8;
    const e = try MMA(&a, &b, 32, alloc);
    defer alloc.free(e);
    assert(e[0] == 26);
    assert(e[16] == 38);
    assert(e[512] == 30);
    assert(e[528] == 44);
}
