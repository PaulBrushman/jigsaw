pub const Precision = enum {
    half,
    byte,

    pub fn max(self: Precision, other: Precision) Precision {
        if (@intFromEnum(self) < @intFromEnum(other)) return self;
        return other;
    }
};
