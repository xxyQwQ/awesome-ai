% Step 1: Blur Image
image = double(imread("baboon.bmp")) / 255; % read image
psf = 0.04 * ones(5); % generate PSF
blurred = conv2(image, psf); % blur by convolution

imwrite(image, "original.jpg"); subplot(4, 4, 1); imshow(image); title("original image");
imwrite(blurred, "blurred.jpg"); subplot(4, 4, 2); imshow(blurred); title("blurred image");



% Step 2: Insert Guassian Noise
noised_30db = awgn(blurred, 30); % SNR = 30dB
noised_20db = awgn(blurred, 20); % SNR = 20dB
noised_10db = awgn(blurred, 10); % SNR = 10dB

imwrite(noised_30db, "noised_30db.jpg"); subplot(4, 4, 5); imshow(noised_30db); title("blurred and noised in 30 dB");
imwrite(noised_20db, "noised_20db.jpg"); subplot(4, 4, 6); imshow(noised_20db); title("blurred and noised in 20 dB");
imwrite(noised_10db, "noised_10db.jpg"); subplot(4, 4, 7); imshow(noised_10db); title("blurred and noised in 10 dB");



% Step 3: Restore Image

% Inverse Filtering
psf = 0.04 * ones(5); psf(516, 516) = 0; PSF = fft2(psf); % fourier transform of PSF
inverse_blurred = ifft2(fft2(blurred) ./ PSF); % blurred case
inverse_noised_30db = ifft2(fft2(noised_30db) ./ PSF); % blurred and noised in 30 dB case
inverse_noised_20db = ifft2(fft2(noised_20db) ./ PSF); % blurred and noised in 20 dB case
inverse_noised_10db = ifft2(fft2(noised_10db) ./ PSF); % blurred and noised in 10 dB case

imwrite(inverse_blurred, "inverse_blurred.jpg"); subplot(4, 4, 9); imshow(inverse_blurred); title("inverse: blurred");
imwrite(inverse_noised_30db, "inverse_noised_30db.jpg"); subplot(4, 4, 10); imshow(inverse_noised_30db); title("inverse: blurred and noised in 30 dB");
imwrite(inverse_noised_20db, "inverse_noised_20db.jpg"); subplot(4, 4, 11); imshow(inverse_noised_20db); title("inverse: blurred and noised in 20 dB");
imwrite(inverse_noised_10db, "inverse_noised_10db.jpg"); subplot(4, 4, 12); imshow(inverse_noised_10db); title("inverse: blurred and noised in 10 dB");

% Wiener Filtering
psf = 0.04 * ones(5); % generate PSF
var_image = var(image(:)); % variance of original image
wiener_blurred = deconvwnr(blurred, psf, var(blurred(:)) / var_image); % blurred case
wiener_noised_30db = deconvwnr(noised_30db, psf, var(noised_30db(:)) / var_image); % blurred and noised in 30 dB case
wiener_noised_20db = deconvwnr(noised_20db, psf, var(noised_20db(:)) / var_image); % blurred and noised in 20 dB case
wiener_noised_10db = deconvwnr(noised_10db, psf, var(noised_10db(:)) / var_image); % blurred and noised in 10 dB case

imwrite(wiener_blurred, "wiener_blurred.jpg"); subplot(4, 4, 13); imshow(wiener_blurred); title("wiener: blurred");
imwrite(wiener_noised_30db, "wiener_noised_30db.jpg"); subplot(4, 4, 14); imshow(wiener_noised_30db); title("wiener: blurred and noised in 30 dB");
imwrite(wiener_noised_20db, "wiener_noised_20db.jpg"); subplot(4, 4, 15); imshow(wiener_noised_20db); title("wiener: blurred and noised in 20 dB");
imwrite(wiener_noised_10db, "wiener_noised_10db.jpg"); subplot(4, 4, 16); imshow(wiener_noised_10db); title("wiener: blurred and noised in 10 dB");

% Rescaling
rescaled_wiener_blurred = wiener_blurred / max(wiener_blurred(:)); imwrite(rescaled_wiener_blurred, "rescaled_wiener_blurred.jpg");
rescaled_wiener_noised_30db = wiener_noised_30db / max(wiener_noised_30db(:)); imwrite(rescaled_wiener_noised_30db, "rescaled_wiener_noised_30db.jpg");
rescaled_wiener_noised_20db = wiener_noised_20db / max(wiener_noised_20db(:)); imwrite(rescaled_wiener_noised_20db, "rescaled_wiener_noised_20db.jpg");
rescaled_wiener_noised_10db = wiener_noised_10db / max(wiener_noised_10db(:)); imwrite(rescaled_wiener_noised_20db, "rescaled_wiener_noised_10db.jpg");