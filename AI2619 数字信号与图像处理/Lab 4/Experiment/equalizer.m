function [filter] = equalizer(freq, coef)
filter = ones(size(freq));
freq = abs(freq);
for i = 1: length(freq)
    if 20 <= freq(i) && freq(i) < 40
        filter(i) = coef(1);
    elseif 40 <= freq(i) && freq(i) < 150
        filter(i) = coef(2);
    elseif 150 <= freq(i) && freq(i) < 500
        filter(i) = coef(3);
    elseif 500 <= freq(i) && freq(i) < 2000
        filter(i) = coef(4);
    elseif 2000 <= freq(i) && freq(i) < 5000
        filter(i) = coef(5);
    elseif 5000 <= freq(i) && freq(i) < 8000
        filter(i) = coef(6);
    elseif 8000 <= freq(i) && freq(i) < 20000
        filter(i) = coef(7);
    end
end