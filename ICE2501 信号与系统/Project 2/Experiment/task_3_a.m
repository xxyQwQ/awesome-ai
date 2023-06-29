load('lineup.mat');
input = y3;
last = length(input);
bias = -last : last;
relevance = self_correlation(input);
plot(bias, relevance);