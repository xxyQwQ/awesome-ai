% Step 0
frame = 0.001; % frame time: 0.001s
time = -5: frame: 15; % time sequence
period = 500; % sample period: 500 frames
sample = 1: period: length(time); % sample position
precision = 4096; % transform precision

% Step 1
signal = zeros(size(time)); % construct signal
for i = 1: length(time)
    if 0 <= time(i) && time(i) <= 10
        signal(i) = 1;
    end
end
point = time(sample); % sample point
result = signal(sample); % sample result
frequency = (0: 4095) / 1000; % frequency point
magnitude = abs(fft(result, 4096)); % frequency result
subplot(4, 2, 1); plot(point, result); xlabel("t/s"); ylabel("x(t)"); grid on;
subplot(4, 2, 2); plot(frequency, magnitude); xlabel("f/Hz"); ylabel("|F(f)|"); grid on;

% Step 2
signal = zeros(size(time)); % construct signal
for i = 1: length(time)
    if 0.5 * period * frame <= time(i) && time(i) <= 10 + 0.5 * period * frame
        signal(i) = 1;
    end
end
point = time(sample); % sample point
result = signal(sample); % sample result
frequency = (0: 4095) / 1000; % frequency point
magnitude = abs(fft(result, 4096)); % frequency spectrum
subplot(4, 2, 3); plot(point, result); xlabel("t/s"); ylabel("x(t)"); grid on;
subplot(4, 2, 4); plot(frequency, magnitude); xlabel("f/Hz"); ylabel("|F(f)|"); grid on;

% Step 3
signal = zeros(size(time)); % construct signal
for i = 1: length(time)
    if 0.5 * period * frame <= time(i) && time(i) <= 10 + 0.5 * period * frame
        signal(i) = 1;
    end
end
[p, q] = butter(2, 0.0005); % butterworth design
signal = filter(p, q, signal); % low-pass filter
point = time(sample); % sample point
result = signal(sample); % sample result
frequency = (0: 4095) / 1000; % frequency point
magnitude = abs(fft(result, 4096)); % frequency spectrum
subplot(4, 2, 5); plot(point, result); xlabel("t/s"); ylabel("x(t)"); grid on;
subplot(4, 2, 6); plot(frequency, magnitude); xlabel("f/Hz"); ylabel("|F(f)|"); grid on;