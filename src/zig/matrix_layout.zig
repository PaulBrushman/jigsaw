pub const Precision = enum {
    half,
    byte,

    pub fn max(self: Precision, other: Precision) Precision {
        if (@intFromEnum(self) < @intFromEnum(other)) return self;
        return other;
    }
};

pub const MatrixLayout = struct {
    const n_precisions = @typeInfo(Precision).@"enum".fields.len;
    x_layout: [n_precisions]usize,
    y_layout: [n_precisions]usize,

    pub fn max_prec(dim: usize) MatrixLayout {
        const layout = [0]usize{0} ** n_precisions;
        layout[0] = dim;
        return MatrixLayout.symetric(layout);
    }

    pub fn symetric(layout: [n_precisions]usize) MatrixLayout {
        return MatrixLayout{ .x_layout = layout, .y_layout = layout };
    }

    pub fn get_precision(self: MatrixLayout, x: usize, y: usize) Precision {
        var x_max = 0;
        var x_prec = 0;
        var y_max = 0;
        var y_prec = 0;

        for (self.x_layout, 0..) |x_block, i|
            if (x_max + x_block > x) {
                x_prec = i;
                break;
            } else {
                x_max += x_block;
            };

        for (self.y_layout, 0..) |y_block, i|
            if (y_max + y_block > y) {
                y_prec = i;
                break;
            } else {
                y_max += y_block;
            };

        if (x_prec > y_prec) return @enumFromInt(x_prec);
        return @enumFromInt(y_prec);
    }
};
