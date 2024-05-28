function [ theta1,theta2,Z_output,beta1,beta2] = isa_gaus_MCP( X,Y,Z_cov,gamma)
%covariate matrix Z
%first response X, seconde response Y
[n,p] = size(Z_cov);

B1= lasso(Z_cov,X,'Alpha',1);
% idxLambda1SE1 = FitInfo1.IndexMinMSE;
beta10 = B1(:,1);
% [B2,FitInfo2]= lasso(Z_cov_bar,Y,'Alpha',1,'CV',10);
% idxLambda1SE2 = FitInfo2.IndexMinMSE;
B2= lasso(Z_cov,Y,'Alpha',1);
beta20 = B2(:,1);

% c=0.5;
% gamma20 = 0.1;
% gamma10_list = [0.001 0.1 0.5 1];
% BIC_gamma10 = BIC_gaus_MCP_gamma1( X,Y,Z_cov,gamma,gamma10_list,gamma20,beta10,beta20,c);
% gamma10 = gamma10_list(find(BIC_gamma10==min(BIC_gamma10)));
% if length(gamma10) > 1
%     gamma10=gamma10(1);
% end
% gamma20_list = [0.01 0.05 0.1 0.2 0.3];
% BIC_gamma20 = BIC_gaus_MCP_gamma2(X,Y,Z_cov,gamma,gamma10,gamma20_list,beta10,beta20,c);
% gamma20 = gamma20_list(find(BIC_gamma20==min(BIC_gamma20)));
% if length(gamma20) > 1
%     gamma20=gamma20(1);
% end
% 
% 
% [~,~,beta1,beta2,~,~] = isa_gaus_MCP_ADMM(X,Y,Z_cov,gamma,gamma10,gamma20,beta10,beta20);
% 
% beta10 = beta1;
% beta20 = beta2;


c=5;
gamma2=0.1;
gamma1_list = [0.6 0.7 0.8 0.9 1 1.1 1.2];  
BIC_gamma1 = BIC_gaus_MCP_gamma1( X,Y,Z_cov,gamma,gamma1_list,gamma2,beta10,beta20,c);
gamma1 = gamma1_list(find(BIC_gamma1==min(BIC_gamma1)));
if length(gamma1) > 1
    gamma1=gamma1(1);
end
% gamma2 = [1 3 5 7 9];
gamma2_list = [0.05 0.1 0.2 0.3];
BIC_gamma2 = BIC_gaus_MCP_gamma2( X,Y,Z_cov,gamma,gamma1,gamma2_list,beta10,beta20,c);
gamma2 = gamma2_list(find(BIC_gamma2==min(BIC_gamma2)));
if length(gamma2) > 1
    gamma2=gamma2(1);
end

[theta1,theta2,beta1,beta2,Z_output,iter] = isa_gaus_MCP_ADMM(X,Y,Z_cov,gamma,gamma1,gamma2,beta10,beta20);



end
