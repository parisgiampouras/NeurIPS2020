%Noisy Matrix Completion Experiment
%Test VarSchatten-p algorithm on different sampling rates and noise levels
%
clc
clear all
warning off



SNR = 10;
missing_rate =[0.2 0.3 0.4 0.5 0.6 0.7 0.8];%


for j=1:7
for nn=1:1
%Dimensions of data matrix 
m=500;
n=500;
r=20;
%
%Generate low-rank matrix
A=randn(m,r);  
B=randn(r,n);
X=A*B;

%
X0= X;
sigma = (((sum((X0(:).^2)))/(m*n)))*10^(-SNR/10)
noise = sqrt(sigma)*randn(m,n); %add noise

X= X + noise; %noisy matrix
SNR=20*log10(norm(X0,'fro')/norm(X-X0,'fro'))


Omega = ones(m,n);
Inds = randperm(m*n,round(m*n*(missing_rate(j))));
Omega(Inds)=0;

X=X.*Omega; %incomplete noisy matrix


%% VarScatten-0.5
p=0.5;
MaxIter = 500;
tol = 1e-4;
threshold = 1e-5; %column pruning threshold
initial_rank = ceil(1.5*r);%ceil(r*0.5);
lambda =0.05*(m+n)*initial_rank; 
rnk_update = 1; %rank-one-activation
 
[U1,V1] = VarSchatten(X,lambda,initial_rank,X0,MaxIter,tol,threshold,p,rnk_update);
%%
X_est{1} = U1*V1';

%% VarSchatten-0.3
p=0.3;

MaxIter = 500;
tol = 1e-4; 
threshold = 1e-5; %column pruning threshold
initial_rank = ceil(1.5*r); %ceil(0.5*r);
lambda = 0.2*(m+n)*initial_rank; 
rnk_update = 1; %rank-one-activation
 
[U2,V2,cost] = VarSchatten(Omega.*X,lambda,initial_rank,X0,MaxIter,tol,threshold,p,rnk_update);
%%
X_est{2} = U2*V2';


%%Relative recovery error
for i=1:2
re_error_M(i,j)=norm((X0-X_est{i}).*(1-Omega),'fro')/norm(X0.*(1-Omega),'fro')
end

end
end


