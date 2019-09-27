clc
clear
%Regression - "Ridge" for 1-D input data

fclose all;
D = 9;                 % Degree of polynomial
jump = 1;               % to skip jump nuumber of degrees
plt = 0;                % to plot no figure (0), input vs. output (1), scatter plot (2)
n_samples = 10;         % number of random samples taken
lambda = 1;

Errors = zeros(D, 2);   % to store the error values
Degree = 1:D;           % for plotting degree of polynomial vs. Error

R = randperm(200);    % generating random numbers

for K = 1:jump:D        % run a loop for 1 to D degrees
N = 1:K;                % used to calculate design matrix           

% reading data from training data file
fileID = fopen('1d_team_13_train.txt','r');
formatSpec = '%f %f';
sizeA = [2 Inf];
A = fscanf(fileID,formatSpec,sizeA);
A =A';

t = A(:, 1)';               % extracting first column - training data input (x1)
x = t(:, R(1:n_samples));   % selecting random n_samples samples

% generating design matrix
X = repmat(x,[length(N)+1 1]).^repmat(([0 N])',[1 length(x)]);
X = X';

t = A(:, 2)';               
T = t(:, R(1:n_samples))';   % extracting second column - target values
w = inv(X'*X + lambda*eye(K+1))*X'*T;         % calculating weight parameters
E = (T - X*w)'*(T - X*w) + lambda*(norm(w)^2);   % calculating training error
Errors(K,1) = E;            % storing training error

%reading data form development data file
fileID = fopen('1d_team_13_dev.txt','r');
formatSpec = '%f %f';
sizeB = [2 Inf];
B = fscanf(fileID,formatSpec,sizeB);
B = B';

t = B(:, 1)';               % extracting first column - development data input (x1)
x = t(:, R(1:n_samples));   % selecting random n_samples samples

%generating design matrix
X1 = repmat(x,[length(N)+1 1]).^repmat(([0 N])',[1 length(x)]);
X1 = X1';
                            
t = B(:, 2)';               
T1 = t(:, R(1:n_samples))';   % extracting second column - target values
Y = X1*w;                   % calculating weight parameters
E1 = (T1 - X1*w)'*(T1 - X1*w);  % calculating development error
Errors(K,2) = E1;           % storing development error


if plt == 1
% Plotting
    figure
    scatter(x', T1, 'filled');
    hold on
    scatter(x', Y, 'filled');
    title(['M = ', num2str(K)]);
    xlabel('Input');
    ylabel('Output');
    legend('Train data', 'Test data');
    hold off
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