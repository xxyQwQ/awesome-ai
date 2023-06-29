function out = single_eliminate(in, period, alpha)
last = length(in);
out = zeros(last, 1);
for k = 0 : last / period
    out = out + (-alpha) ^ k * time_shift(in, -period * k);
end