function K = rbf(X,Y,sigma)
    R = L2_distance(X,Y);
    K = exp(-R.^2/(2*sigma^2));
end