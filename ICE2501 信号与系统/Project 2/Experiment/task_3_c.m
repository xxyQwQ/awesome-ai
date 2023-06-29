load("lineup.mat");
input = y3;
last = length(input);
output = double_eliminate(input, 751, 0.75, 2252, 0.60);
subplot(2, 1, 1);
plot(input);
subplot(2, 1, 2);
plot(output);
sound(output);