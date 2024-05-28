function [BIC_gamma1] = BIC_gaus_MCP_gamma1( X,Y,Z,gamma,gamma1_list,gamma2,beta10,beta20,c)
%%% Calculate BIC for gamma1

[n,p] = size(Z);
alpha_X = 1/ norm(X - mean(X,2),'fro')^2;
alpha_Y = 1/ norm(Y - mean(Y,2),'fro')^2;
alpha2 = 0.5;
alpha_scale = min([alpha_X,alpha_Y]);
alpha_X =  (1-alpha2) * alpha_X / alpha_scale;
alpha_Y =  alpha2 * alpha_Y / alpha_scale;

BIC_gamma1 = zeros(1,length(gamma1_list));

for i = 1:length(gamma1_list)

[theta1,theta2,beta1,beta2,Z_output,~] = isa_gaus_MCP_ADMM(X,Y,Z,gamma,gamma1_list(i),gamma2,beta10,beta20);

[no_class,~] = group_assign_vertice(Z_output,n);

p1 = sum(beta1~=0);
p2 = sum(beta2~=0);

data_fidelity1 = norm(X-theta1-(Z*beta1)','fro')^2;

data_fidelity2 = norm(Y-theta2-(Z*beta2)','fro')^2;

Cn = c*log(log(2*n+2*p));

BIC_gamma1(i) = alpha_X * log(data_fidelity1/n) + alpha_Y * log(data_fidelity2/n) + Cn * log(n) * (2*no_class+p1+p2)/(n);
end

end