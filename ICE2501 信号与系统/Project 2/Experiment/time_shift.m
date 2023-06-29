function out = time_shift(in, bias)
last = length(in);
out = in;
if bias > 0
    for i = 1 : last - bias
        out(i, 1) = in(i + bias, 1);
    end
    for i = last - bias + 1 : last
        out(i, 1) = 0;
    end
end
if bias < 0
    for i = 1 - bias : last
        out(i, 1) = in(i + bias, 1);
    end
    for i = 1 : 1 - bias - 1
        out(i, 1) = 0;
    end
end