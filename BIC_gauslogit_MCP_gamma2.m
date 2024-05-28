function [BIC_gamma2] = BIC_gauslogit_MCP_gamma2( X,Y,Z,gamma,gamma1,gamma2_list,beta10,beta20,c)
%%% Calculate BIC for gamma1
rho=1;
[n,p] = size(Z);

alpha_X = 2 / norm(X - mean(X,2),'fro')^2;
logitbary = log(mean(Y,2)/(1-mean(Y,2)));
logitloss0 = - sum(Y * logitbary) + n * log(1+exp(logitbary));

alpha_Y = 1/ logitloss0;

alpha2 = 0.5;

alpha_scale = min([alpha_X,alpha_Y]);
alpha_X =  (1-alpha2) * alpha_X / alpha_scale;
alpha_Y =  alpha2 * alpha_Y / alpha_scale;


BIC_gamma2 = zeros(1,length(gamma2_list));

for i = 1:length(gamma2_list)
gamma1
gamma2_list(i)
[theta1,theta2,beta1,beta2,Z_output,~] = isa_gauslogit_MCP_ADMM(X,Y,Z,gamma,gamma1,gamma2_list(i),beta10,beta20,rho);
[no_class,~] = group_assign_vertice(Z_output,n);

p1 = sum(beta1~=0);
p2 = sum(beta2~=0);

data_fidelity1 = norm(X-theta1-(Z*beta1)','fro')^2;

data_fidelity2 = - Y * (theta2'+ Z*beta2) + sum(log(1+exp(theta2+(Z*beta2)')));

Cn = c*log(log(2*n+2*p));

BIC_gamma2(i) = alpha_X * log(data_fidelity1/n) + 2 * alpha_Y * data_fidelity2/n + Cn * log(n) * (2*no_class+p1+p2)/n;
end

end