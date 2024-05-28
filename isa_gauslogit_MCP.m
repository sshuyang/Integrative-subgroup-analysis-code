function [ U1,U2,V,beta1,beta2] = isa_gauslogit_MCP( X,Y,Z_cov,gamma )
%covariate matrix Z
%first response X, seconde response Y
%%% Binary Y
rho=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n,p] = size(Z_cov);


% Z_cov_bar=[ones(n,1) Z_cov];
% beta10 = inv(Z_cov_bar'*Z_cov_bar)*Z_cov_bar'*X'; %%initial value: least square estimator 
% beta10 = beta10(2:(p+1));
B1 = lasso(Z_cov,X,'Alpha',1);
beta10 = B1(:,1);
% B2= lassoglm(Z_cov,Y','binomial');
% beta20 = B2(:,1);
beta20 = zeros(p,1);

%%select gamma1 and gamma2
% c=0.5;
% gamma=3;
% gamma10 = 0.7;
% gamma20 = 0.3;
% gamma20 = 0.2;
% gamma10 = [0.3 0.5 0.7 0.9];
% BIC_gamma10 = BIC_gauslogit_MCP_gamma1( X,Y,Z_cov,gamma,gamma10,gamma20,beta10,beta20,c);
% gamma10 = gamma10(find(BIC_gamma10==min(BIC_gamma10)));
% if length(gamma10) > 1
%    gamma10=gamma10(1);
% end
% gamma20 = [0.2 0.3 0.4];
% BIC_gamma20 = BIC_gauslogit_MCP_gamma2( X,Y,Z_cov,gamma,gamma10,gamma20,beta10,beta20,c);
% gamma20 = gamma20(find(BIC_gamma20==min(BIC_gamma20)));
% if length(gamma20) > 1
%     gamma20=gamma20(1);
% end

% [U1,U2,beta1,beta2,V,iter] = isa_gauslogit_MCP_ADMM(X,Y,Z_cov,gamma,gamma10,gamma20,beta10,beta20,rho);

% [no_class,class_id] = group_assign_vertice(V,n);

% beta10=beta1;
% beta20=beta2;
c=5;
gamma2=0.2;
%gamma2=gamma20;
gamma1 = [0.6 0.7 0.8 0.9 1];
BIC_gamma1 = BIC_gauslogit_MCP_gamma1( X,Y,Z_cov,gamma,gamma1,gamma2,beta10,beta20,c);
gamma1 = gamma1(find(BIC_gamma1==min(BIC_gamma1)));
if length(gamma1) > 1
    gamma1=gamma1(1);
end
gamma2 = [0.2 0.3 0.4 0.5];
BIC_gamma2 = BIC_gauslogit_MCP_gamma2( X,Y,Z_cov,gamma,gamma1,gamma2,beta10,beta20,c);
gamma2 = gamma2(find(BIC_gamma2==min(BIC_gamma2)));
if length(gamma2) > 1
    gamma2=gamma2(1);
end

[U1,U2,beta1,beta2,V,iter] = isa_gauslogit_MCP_ADMM(X,Y,Z_cov,gamma,gamma1,gamma2,beta10,beta20,rho);



end