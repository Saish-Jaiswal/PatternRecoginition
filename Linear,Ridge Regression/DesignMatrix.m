function [DM] = DesignMatrix(X,degree)
    [M,N] = size(X);
    DM = zeros(N, degree+1);
    DM(:,1) = 1;
    for i = 1:N
        for j = 1:M
            for k = 1:degree
                DM(i,k+1) = DM(i,k+1) + X(j, i)^k;
            end
        end
    end
end

