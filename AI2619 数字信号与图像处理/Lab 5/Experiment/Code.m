I = double(imread('roman.jpg')) / 255;
R = I(:, :, 1); G = I(:, :, 2); B = I(:, :, 3);
figure(1); imshow(R);
figure(2); imhist(R, 1024);
J = histeq(R);
figure(3); imshow(J);
figure(4); imhist(J, 1024);

F = exp(-0.04 .* (1: 256));
J = histeq(R, F);
figure(5); imshow(J);
figure(6); imhist(J, 1024);
F = exp(-1 .* ((1: 256) / 64) .^ 2);
J = histeq(R, F);
figure(7); imshow(J);
figure(8); imhist(J, 1024);

F = ((1: 256) - 128) .^ 2;
J = histeq(R, F);
figure(9); imshow(J);
figure(10); imhist(J, 1024);

U = histeq(R); V = histeq(G); W = histeq(B);
J(:, :, 1) = U; J(:, :, 2) = V; J(:, :, 3) = W;
figure(11); imshow(J);
figure(12); imhist(J, 1024);

J = histeq(I);
figure(13); imshow(J);
figure(14); imhist(J, 1024);
U = adapthisteq(R); V = adapthisteq(G); W = adapthisteq(B);
J(:, :, 1) = U; J(:, :, 2) = V; J(:, :, 3) = W;
figure(15); imshow(J);
figure(16); imhist(J, 1024);