function out = self_correlation(in)
last = length(in);
out = zeros(2 * last + 1, 1);
for n = -last : last
    shift = time_shift(in, n);
    out(n + last + 1, 1) = dot(in, shift);
end