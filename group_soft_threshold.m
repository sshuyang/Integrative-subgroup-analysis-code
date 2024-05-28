function z = group_soft_threshold(a, kappa)
    if kappa==inf
        z=0;
    else
    tmp = 1 - kappa/norm(a);
    z = (tmp >0) * tmp * a;
    end
end