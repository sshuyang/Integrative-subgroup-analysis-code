function [U1,U2,beta1,beta2,V,iter] = isa_gaus_MCP_ADMM(X,Y,T,gamma,gamma1,gamma2,beta10,beta20)


alpha_X = 1/ norm(X - mean(X,2),'fro')^2;
alpha_Y = 1/ norm(Y - mean(Y,2),'fro')^2;

alpha2 = 0.5;

alpha_scale = min([alpha_X,alpha_Y]);
alpha_X =  (1-alpha2) * alpha_X / alpha_scale;
alpha_Y =  alpha2 * alpha_Y / alpha_scale;


[p1,n] = size(X);
[p2,n] = size(Y);
[n,p_cov] = size(T);

[x,y] = meshgrid(1:n, 1:n);
A = [x(:) y(:)];

A = A(y(:)>x(:),:);

[len_l,~] = size(A);

l1_mat_org = zeros(len_l,n);
l2_mat_org = zeros(len_l,n);

for i = 1:n
    l1_mat_org(:,i) = (A(:,1) == i);
end

for i = 1:n
    l2_mat_org(:,i) = (A(:,2) == i);
end

D = l1_mat_org - l2_mat_org;
D = D';

MAX_ITER  = 500;
INNER_ITER = 50;

beta1 = beta10; %inv(T'* T)*T'*X';
beta2 = beta20; %inv(T'* T)*T'*Y';

% beta1 = zeros(p_cov,1);
% beta2 = zeros(p_cov,1);

U1 = X-(T * beta1)';
U2 = Y-(T * beta2)';

V1 = U1 * D;
V2 = U2 * D;
V = [V1;V2];

Lambda1 = zeros(p1,len_l);
Lambda2 = zeros(p2,len_l);

M1 = inv(alpha_X * eye(n) + D * D');
M2 = inv(alpha_Y * eye(n) + D * D');

beta = zeros(2*p_cov,INNER_ITER);
% U_output = zeros(p1+p2,n,MAX_ITER);
% Z_output = zeros(p1+p2,len_l,MAX_ITER);
% beta1_output = zeros(p_cov,MAX_ITER);
% beta2_output = zeros(p_cov,MAX_ITER);
% 
%  U_output(:,:,1) = [U;V];  
%  beta1_output(:,1) = beta1;
%  beta2_output(:,1) = beta2;
%  Z_output(:,:,1) = Z;
%obj = zeros(1,MAX_ITER);
%time_per_iter = zeros(1,MAX_ITER);
iter=0;

for m = 1:MAX_ITER

        beta(:,1,m) = [beta1;beta2];
    
    for k = 1:INNER_ITER
%tic 
U1 = (alpha_X * (X - beta1'*T')  + (V1+Lambda1) * D') * M1;
U2 = (alpha_Y * (Y - beta2'*T') + (V2+Lambda2) * D') * M2;
        
%         for j = 1:p_cov
%         beta1(j) = group_soft_threshold( (X- U1- beta1'*T'+beta1(j)*T(:,j)')*T(:,j) , gamma2*elta1(j)/alpha_X) / (T(:,j)'*T(:,j));
%         beta2(j) = group_soft_threshold( (Y- U2- beta2'*T'+beta2(j)*T(:,j)')*T(:,j) , gamma2*elta2(j)/alpha_Y) / (T(:,j)'*T(:,j));
%         end

%          beta1_ls = inv(T'*T)*T'*(X- U1)';
%         beta2_ls = inv(T'*T)*T'*(Y- U2)';       
%     for j = 1:p_cov
%             if norm(beta1_ls(j))<=gamma*gamma2
%         beta1(j) = gamma*group_soft_threshold(beta1_ls(j),gamma2)/(gamma-1);
%             else beta1(j) = beta1_ls(j);
%             end
%         if norm(beta2_ls(j))<=gamma*gamma2
%        beta2(j) = gamma*group_soft_threshold(beta2_ls(j),gamma2)/(gamma-1);
%             else beta2(j) = beta2_ls(j);
%         end
%     end
model1 = glm_gaussian((X-U1)',T,'nointercept');
model2 = glm_gaussian((Y-U2)',T,'nointercept');
fit1 = penalized(model1,@p_mcplus,'w',gamma,'scaled',true,'lambda',gamma2);
fit2 = penalized(model2,@p_mcplus,'w',gamma,'scaled',true,'lambda',gamma2);
beta1 = fit1.beta;
beta2 = fit2.beta;

        beta(:,k+1,m) = [beta1;beta2];
      
        if norm(beta(:,k+1,m)-beta(:,k,m)) < 1e-6
           break
        end
    end
      

 for l = 1:len_l
        tmp = A(l,1);
        tmp2 = A(l,2);
        
        V1(l) = U1(tmp) - U1(tmp2) - Lambda1(l);
        V2(l) = U2(tmp) - U2(tmp2) - Lambda2(l);
        
        if norm([V1(l);V2(l)]) <= gamma * gamma1
        V(:,l) = gamma * group_soft_threshold([V1(l);V2(l)], gamma1) / (gamma-1);
        else
            V(:,l) = [V1(l);V2(l)];
        end
       
        V1(l) = V(1,l);
        V2(l) = V(2,l);
        
        Lambda1(l) = Lambda1(l) + (V1(l) - U1(tmp) + U1(tmp2));
        Lambda2(l) = Lambda2(l) + (V2(l) - U2(tmp) + U2(tmp2));
        
 end
% 
%  U_output(:,:,m+1) = [U;V];  
%  beta1_output(:,m+1) = beta1;
%  beta2_output(:,m+1) = beta2;
%  Z_output(:,:,m+1) = Z;
    %time_per_iter(m) = toc;

    iter=iter+1;
    if m > 10 && norm([U1;U2]*D-V,'fro') < 1e-5
        break
    end

end
