function [U1,U2,beta1,beta2,V,iter]=isa_gauslogit_MCP_ADMM(X,Y,T,gamma,gamma1,gamma2,beta10,beta20,rho)

[n,p_cov] = size(T);

alpha_X = 2 / norm(X - mean(X,2),'fro')^2;
logitbary = log(mean(Y,2)/(1-mean(Y,2)));
logitloss0 = - sum(Y * logitbary) + n * log(1+exp(logitbary));

alpha_Y = 1/ logitloss0;

alpha2 = 0.5;

alpha_scale = min([alpha_X,alpha_Y]);
alpha_X =  (1-alpha2) * alpha_X / alpha_scale;
alpha_Y =  alpha2 * alpha_Y / alpha_scale;



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
INNER_ITER = 10;

%set initial values

beta1 = beta10; %inv(T'* T)*T'*X';
beta2 = beta20;
beta = zeros(2*p_cov,INNER_ITER);

% beta1 = zeros(p_cov,1);
% beta2 = zeros(p_cov,1);

U1 = X-(T * beta1)';
U2 = zeros(1,n);
     P = exp(T*beta2)./(1+exp(T*beta2));
% Y_bar = Y;
% Y_bar(Y==0)=0.5;
% Y_bar(Y==1)=0.5;
% temp = log(Y_bar./(1-Y_bar));
%P = (exp(temp)./(1+exp(temp)))';
      W = diag(P .* (1-P));
    M2 = inv(alpha_Y * W + rho * (D * D'));
  Z = (T*beta2 + (Y'-P)./ (P.*(1-P)))';
% Z = (temp' + (Y'-P)./ (P.*(1-P)))';
%U2 = (Z - beta2'*T') * W ;
%U2 = temp - (T*beta2)';


V1 = U1 * D;
V2 = U2 * D;
V = [V1;V2];

Lambda1 = zeros(1,len_l);
Lambda2 = zeros(1,len_l);

M1 = inv(alpha_X * eye(n) + rho * (D * D'));
% W = 1/4 * eye(n);
% M2 = inv(alpha_Y * 1/4 * eye(n) + rho * D * D');
% P = exp(U2')./(1+exp(U2'));
% Z = (U2'+ (Y'-P)./ (P.*(1-P)))';


 %U_output = zeros(2,n,MAX_ITER);
 %V_output = zeros(2,len_l,MAX_ITER);
 %Lambda_output = zeros(2,len_l,MAX_ITER);
 %beta_output = zeros(2*p_cov,MAX_ITER);
 
 
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
 beta(:,1)=[beta1;beta2];
  
 for k=1:INNER_ITER
  U1 = (alpha_X * (X - beta1'*T')  + rho * (V1 + Lambda1/rho) * D') * M1;
  
  U2 = (alpha_Y * (Z - beta2'*T') * W + rho * (V2 + Lambda2/rho) * D') * M2;
%    for h=1:INNER_ITER
%        U2_tem(h+1,:) = U2_tem(h,:) - (alpha_Y * (P' - Y) + (U2_tem(h,:) * D - V2 - Lambda2) *D') * M2;
%   P = exp(U2_tem(h+1,:)'+T*beta2)./(1+exp(U2_tem(h+1,:)'+T*beta2));
%        W1 = diag(P .* (1-P));
%     M2 = inv(alpha_Y * W1 + D * D');
% 
% if norm(U2_tem(h+1,:)-U2_tem(h,:)) < 1e-6
%     U2=U2_tem(h+1,:);
%     break
% end
%     end
     
    model1 = glm_gaussian((X-U1)',T,'nointercept');
    fit1 = penalized(model1,@p_mcplus,'w',gamma,'scaled',true,'lambda',gamma2);
    beta1 = fit1.beta;  
for j = 1:p_cov
 if norm((Z - U2 - beta2'*T' + beta2(j)*T(:,j)') * W * T(:,j)) <= gamma * gamma2 * T(:,j)'* W * T(:,j) 
%   beta1(j) = group_soft_threshold( (X - U1 - beta1'*T'+beta1(j)*T(:,j)')*T(:,j) , gamma2*elta1(j)/alpha_X) / (T(:,j)'*T(:,j));
        beta2(j) = gamma * group_soft_threshold( (Z - U2 - beta2'*T'+beta2(j)*T(:,j)') * W * T(:,j) / n , gamma2) / (gamma * T(:,j)'* W * T(:,j)/n - 1 ); 

 else 
     beta2(j) = (Z - U2 - beta2'*T'+beta2(j)*T(:,j)') * W * T(:,j) / (T(:,j)'* W * T(:,j));
     
 end
end

  beta(:,k+1)= [beta1;beta2];
   if norm(beta(:,k+1)-beta(:,k)) < 1e-6
       break
   end
%    U2_tem(1,:)=U2;
 end

     P = exp(U2'+T*beta2)./(1+exp(U2'+T*beta2));
      W = diag(P .* (1-P));
    M2 = inv(alpha_Y * W + rho * D * D');
 Z = (Y'-P)./ (P.*(1-P));
 Z(isnan(Z))=1; 
 Z = (Z + U2'+ T*beta2)'; 
 
 for l = 1:len_l
        tmp = A(l,1);
        tmp2 = A(l,2);
        
        V1(l) = U1(tmp) - U1(tmp2) - Lambda1(l)/rho;
        V2(l) = U2(tmp) - U2(tmp2) - Lambda2(l)/rho;
        
        if norm([V1(l);V2(l)]) <= gamma * gamma1
        V(:,l) = gamma * rho * group_soft_threshold([V1(l);V2(l)], gamma1/rho) / (rho * gamma - 1);
        else
            V(:,l) = [V1(l);V2(l)];
        end
       
        V1(l) = V(1,l);
        V2(l) = V(2,l);
        
        Lambda1(l) = Lambda1(l) + rho * (V1(l) - U1(tmp) + U1(tmp2));
        Lambda2(l) = Lambda2(l) + rho * (V2(l) - U2(tmp) + U2(tmp2));
        
 end
% 
     %U_output(:,:,m) = [U1;U2];  
  %beta_output(:,m) = [beta1;beta2];
  %V_output(:,:,m)=[V1;V2];
  %Lambda_output(:,:,m)=[Lambda1;Lambda2];
  
%  beta1_output(:,m+1) = beta1;
%  beta2_output(:,m+1) = beta2;
%  Z_output(:,:,m+1) = Z;
    %time_per_iter(m) = toc;

    iter=iter+1;
    if m > 10 && norm([U1;U2]*D-V,'fro') < 1e-5
        break
    end

 end


