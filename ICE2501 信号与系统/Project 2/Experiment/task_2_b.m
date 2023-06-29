load('lineup.mat');
input = y2;
last = 7000;
period = 501;

total = floor(last / period);
coef = zeros(1, total);
best_alpha = 0;
best_error = 1e10;

for step = 0 : 100
    alpha = 0.01 * step;
    error = 0;

    capacity = last - (total - 1) * period;
    for bias = 1 : capacity
        for k = 1 : total
            coef(1, k) = (-1) ^ (total - k) * input((k - 1) * period + bias, 1);
        end
        error = error + abs(polyval(coef, alpha));
    end
    error = error / capacity;

    if error < best_error
        best_alpha = alpha;
        best_error = error;
    end
end

disp(best_alpha);