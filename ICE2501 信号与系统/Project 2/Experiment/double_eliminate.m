function out = double_eliminate(in, period_0, alpha_0, period_1, alpha_1)
last = length(in);
out = zeros(last, 1);
for t = 1 : last
    if t <= period_0
        out(t, 1) = in(t, 1);
    elseif period_0 < t && t <= period_1
        out(t, 1) = in(t, 1) - alpha_0 * out(t - period_0, 1);
    else
        out(t, 1) = in(t, 1) - alpha_0 * out(t - period_0, 1) - alpha_1 * out(t - period_1, 1);
    end
end