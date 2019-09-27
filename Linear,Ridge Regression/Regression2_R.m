clc
clear
%Regression - "Ridge" for 2-D input data

fclose all;
D = 9;                 % Degree of polynomial
jump = 1;               % to skip jump nuumber of degrees
plt = 0;                % to plot no figure (0), input vs. output (1), scatter plot (2)
n_samples = 10;        % number of random samples taken
lambda = 1;          % lambda parameter

Errors = zeros(D, 2);   % to store the error values
Degree = 1:D;           % for plotting degree of polynomial vs. Error

R = randperm(1000);    % generating random numbers

for K = 1:jump:D        % run a loop for 1 to D degrees
N = 1:K;                % used to calculate design matrix

% reading data from training data file
fileID = fopen('2d_team_13_train.txt','r');
formatSpec = '%f %f';
sizeA = [3 Inf];
A = fscanf(fileID,formatSpec,sizeA);
A =A';

t = A(:, 1:2)';               % extracting first column - training data input (x1)
x_temp = t(:, R(1:n_samples));   % selecting random n_samples samples
X = DesignMatrix(x_temp,K); % creating design matrix

t = A(:, 3)';               
T = t(:, R(1:n_samples))';   % extracting thing column - target values
w = inv(X'*X + lambda*eye(K+1))*X'*T;         % calculating weight parameters
E = (T - X*w)'*(T - X*w) + lambda*(norm(w)^2);   % calculating training error
Errors(K,1) = E;            % storing training error

%reading data form development data file
fileID = fopen('2d_team_13_dev.txt','r');
formatSpec = '%f %f';
sizeB = [3 Inf];
B = fscanf(fileID,formatSpec,sizeB);
B = B';

t = B(:, 1:2)';               % extracting first and second column - development data input (x1)
x_temp_dev = t(:, R(1:n_samples));   % selecting random n_samples samples
X1 = DesignMatrix(x_temp_dev,K);    %creating design matrix

t = B(:, 3)';               
T1 = t(:, R(1:n_samples))';   % extracting third column - target values
Y = X1*w;                   % calculating output using trained parameters
E1 = (T1 - Y)'*(T1 - Y);    % calculating development error
Errors(K,2) = E1;           % storing development error

if plt == 1
    % Plotting data
    figure
    title('degree - ' + K);
    scatter3(x_temp_dev(1,:)', x_temp_dev(2,:)', T1, 'filled');  % development inputs (x1, x2) vs. development target
    hold on
    scatter3(x_temp_dev(1,:)', x_temp_dev(2,:)', Y, 'filled');   % development inputs (x1, x2) vs. output
    title(['M = ', num2str(K)]);
    xlabel('Input 1');
    ylabel('Input 2');
    zlabel('Output');
    legend('Train data', 'Test Data');
    hold off;
end

if plt == 2
    figure
    scatter(T1, Y, 'filled');
    title(['M = ', num2str(K)]);
    xlabel('Target Output');
    ylabel('Model Output');
end

end

% Plotting degree vs. error
subplot(1,2,1)
plot(Degree, Errors(:,1));  % training error
xlabel('Model Complexity - Degree of Polynomial');
ylabel('Training Error');
subplot(1,2,2)
plot(Degree, Errors(:,2));  % error while testing with development data
xlabel('Model Complexity - Degree of Polynomial');
ylabel('Testing Error');

[minTrainError,indexMinTrainError] = min(Errors(:,1));  % minimum training error with corresponding degree
[minTestError,indexMinTestError] = min(Errors(:,2));    % minimum testing error with corresponding degree