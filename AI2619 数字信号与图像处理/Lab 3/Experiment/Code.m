% method 1: naive iteration
rng(0); % random seed
length_iteration = 2 .^ (0: 4: 12); % length record
time_iteration = zeros(length(length_iteration), 1); % time record
for i = 1: length(length_iteration)
    N = length_iteration(i); % vector length
    x = rand(N, 1); % input vector
    profile on; % start record
    X = zeros(N, 1); % output vector
    W = exp(-1j * 2 * pi * (0: N-1) / N); % unit root
    for k = 1: N
        for n = 1: N
            X(k) = X(k) + x(n) * (W(n) ^ k); % accumulate term
        end
    end
    P = profile('info'); % stop record
    time_iteration(i) = P.FunctionTable.TotalTime; % record time
end
length_iteration(1) = []; % kick out initial result
time_iteration(1) = []; % kick out initial result

% method 2: matrix computation
rng(0); % random seed
length_matrix = 2 .^ (0: 4: 12); % length record
time_matrix = zeros(length(length_matrix), 1); % time record
for i = 1: length(length_matrix)
    N = length_matrix(i); % vector length
    x = rand(N, 1); % input vector
    profile on; % start record
    W = exp(-1j * 2 * pi * (0: N-1) / N); % unit root
    F = fliplr(vander(W)); % vandermonde matrix
    X = F * x; % matrix computation
    P = profile('info'); % stop record
    time_matrix(i) = P.FunctionTable.TotalTime; % record time
end
length_matrix(1) = []; % kick out initial result
time_matrix(1) = []; % kick out initial result

% method 3: builtin function
rng(0); % random seed
length_builtin = 2 .^ (0: 4: 24); % length record
time_builtin = zeros(length(length_builtin), 1); % time record
for i = 1: length(length_builtin)
    N = length_builtin(i); % vector length
    x = rand(N, 1); % input vector
    profile on; % start record
    X = fft(x, N); % builtin function
    P = profile('info'); % stop record
    time_builtin(i) = P.FunctionTable.TotalTime; % record time
end
length_builtin(1) = []; % kick out initial result
time_builtin(1) = []; % kick out initial result

% method 4: gpu acceleration
rng(0); % random seed
length_gpu = 2 .^ (0: 4: 24); % length record
time_gpu = zeros(length(length_gpu), 1); % time record
for i = 1: length(length_gpu)
    N = length_gpu(i); % vector length
    x = gpuArray(rand(N, 1)); % input vector
    profile on; % start record
    X = fft(x); % gpu calculation
    P = profile('info'); % stop record
    time_gpu(i) = P.FunctionTable.TotalTime; % record time
end
length_gpu(1) = []; % kick out initial result
time_gpu(1) = []; % kick out initial result

% plot result
curve_iteration = plot(log2(length_iteration), log10(time_iteration)); hold on;
curve_matrix = plot(log2(length_matrix), log10(time_matrix)); hold on;
curve_builtin = plot(log2(length_builtin), log10(time_builtin)); hold on;
curve_gpu = plot(log2(length_gpu), log10(time_gpu)); hold on;
legend([curve_iteration, curve_matrix, curve_builtin, curve_gpu], 'Iteration', 'Matrix', 'Builtin', 'GPU');
xlabel('N (log scale)'); ylabel('T (log scale)'); title('Running Time'); grid on;