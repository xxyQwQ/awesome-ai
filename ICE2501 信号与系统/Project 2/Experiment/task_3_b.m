load("lineup.mat");
input = y3;
last = length(input);
period_0 = 751;
period_1 = 2252;

best_alpha_0 = 0;
best_alpha_1 = 0;
best_error = 1e10;

for i = 0 : 100
    for j = 0 : 100
        alpha_0 = 0.01 * i;
        alpha_1 = 0.01 * j;

        output = double_eliminate(input, period_0, alpha_0, period_1, alpha_1);
        error = 0;
        for k = 6000 : 7000
            error = error + abs(output(k, 1));
        end

        if error < best_error
            best_alpha_0 = alpha_0;
            best_alpha_1 = alpha_1;
            best_error = error;
        end
    end
end

disp(best_alpha_0);
disp(best_alpha_1);