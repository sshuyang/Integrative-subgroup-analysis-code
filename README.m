
%%% Case1
p=5;
n=120;
G=3;
gamma=3; %MCP parameter
% c2=5;
beta_1=[1,1,1,1,1,zeros(1,p-5)]'; % True covariate effect
beta_2=[1,1,1,1,1,zeros(1,p-5)]';
mu_1=[repmat(-2,1,n/G),repmat(0,1,n/G),repmat(2,1,n/G)]; %group center
mu_2=[repmat(-2,1,n/G),repmat(0,1,n/G),repmat(2,1,n/G)];


mu=zeros(p,1);
Z=mvnrnd(mu,eye(p),n);
Z=zscore(Z);
tem1=Z*beta_1;
tem2=Z*beta_2;
eps11=normrnd(0,0.5,1,n/G);
eps12=normrnd(0,0.5,1,n/G);
eps13=normrnd(0,0.5,1,n/G);
eps1=[eps11 eps12 eps13];
eps21=normrnd(0,0.5,1,n/G);
eps22=normrnd(0,0.5,1,n/G);
eps23=normrnd(0,0.5,1,n/G);
eps2=[eps21 eps22 eps23];
X=mu_1+tem1'+eps1; %First response
Y=mu_2+tem2'+eps2; %Second response


[theta1,theta2,V,beta1,beta2] =isa_gaus_MCP( X,Y,Z,gamma );

[no_class,class_id] = group_assign_vertice(V,n);
q1 = sum(beta1~=0);
q2 = sum(beta2~=0);
RI = RandIndex(class_id,class_id_ture);





%%% Case2
p=7;
n=120;
G=3;
gamma=3; %MCP parameter
% c2=5;
mu=zeros(p,1);
% beta_1=[1,1,1,1,1,zeros(1,p-5)]';
% beta_2=[1,1,1,1,1,1,1,zeros(1,p-7)]';
beta_1 = [1,1,1,1,1,0,0];
beta_2 = [1,1,1,1,1,1,1];
mu_1=[repmat(1,1,n/G),repmat(-2,1,n/G),repmat(2,1,n/G)];
mu_2=[repmat(-2,1,n/G),repmat(0,1,n/G),repmat(2,1,n/G)];

Z=mvnrnd(mu,eye(p),n);
Z=zscore(Z);
tem1=Z*beta_1;
tem2=Z*beta_2;
eps11=normrnd(0,0.5,1,n/G);
eps12=normrnd(0,0.5,1,n/G);
eps13=normrnd(0,0.5,1,n/G);
eps1=[eps11 eps12 eps13];
eps21=normrnd(0,0.5,1,n/G);
eps22=normrnd(0,1,1,n/G);
eps23=normrnd(0,0.5,1,n/G);
eps2=[eps21 eps22 eps23];
X=mu_1+tem1'+eps1;
Y=mu_2+tem2'+eps2;


[ theta1,theta2,V,beta1,beta2] =isa_gaus_MCP( X,Y,Z,gamma);



[no_class,class_id] = group_assign_vertice(Z_output,n);


%%% Case3
G=2;
p=50;
n=100;
mu=zeros(p,1);
class_id_ture = repelem([1,2],n/G);
beta_1=[1,1,1,1,1,zeros(1,p-5)]';
beta_2=[1,1,1,1,1,zeros(1,p-5)]';
alpha_1=[repmat(-2,1,n/G),repmat(2,1,n/G)];
alpha_2=[repmat(-4,1,n/G),repmat(4,1,n/G)];
Z=mvnrnd(mu,eye(p),n);
Z=zscore(Z);
mu_1=[repmat(-2,1,n/G),repmat(2,1,n/G)];
mu_2=exp(alpha_2+(Z*beta_2)')./(1+exp(alpha_2+(Z*beta_2)'));
tem1=Z*beta_1;
eps1=normrnd(0,0.5,1,n);
X=mu_1+tem1'+eps1;
Y=binornd(1,mu_2,1,n);
RI2_gauslogit(m) = RandIndex(Y,class_id_ture);
[ theta1,theta2,V,beta1,beta2]  = isa_gauslogit_MCP( X,Y,Z,3);




