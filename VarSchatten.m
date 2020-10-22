function [V,U,NMAE,NRE] = VarSchatten(Y,lambda,L,Ytrue,Max_Iter,tol,threshold,p,rank_upd);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs
% Y: mxn incomplete data matrix 
% lambda : low-rank regularization parameter
% Y_true: true matrix or matrix containing test set for evaluating the
% performance of the algorithm
% Max_Iter :  Maximum number of iterations
% threshold : used for stopping the algorithm 

%Outputs 
% V : estimated V matrix
% U : estimated U matrix 
% NMAE : Normalized mean average error -(used for testing performance on movielens dataset)
% NRE : Norlized recontruction error
% 
%
%Code for VarSchatten algorithm associated with the paper:
% P. Giampouras, R. Vidal, A. Rontogiannis, B. Haeffele, "A novel
% variational form of the Schatten-p quasi-norm", Neurips 2020.
% Written by Paris Giampouras, 10/2020 e-mail: parisg@jhu.edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
%Initialization process--------------------------------------------------
d = 1*ones(L,1);
[m,n] = size(Y);

%SVD initialization
%disp('SVD initialization....');
%[u,s,v] =svd(Y);
%U = u(:,1:L)*s(1:L,1:L)^p;
%V = v(:,1:L)*s(1:L,1:L)^(1-p);

%Random initialization
disp('Random Initialization...');
U  = 10*randn(m,L); 
V  = Y'*U*inv(U'*U + lambda*eye(L));

for l=1:L
           d(l) = ((U(:,l)'*U(:,l) + V(:,l)'*V(:,l) ) + eps)^(1-p);
 end
        D = diag((1/2^p)*p*lambda.*(d.^(-1)));
%--------------------------------------------------------------------------
[indx,indy,Ys] = find(Y); %indx, indy correspond to indexes of known entries
relative_error = zeros(Max_Iter,1);
telapsed = zeros(Max_Iter,1);
cost_iter = zeros(Max_Iter,1);
NMAE = zeros(Max_Iter,1);
rel_obj = zeros(Max_Iter,1);

[indx_true,indy_true] = find(Ytrue); 
S = sum(U(indx_true,:).*V(indy_true,:),2);
UV_true = sparse(indx_true,indy_true,S,m,n);
cost_iter(1) = cost(Ytrue,UV_true,d,lambda,p);
Y = sparse(indx,indy,Ys,m,n);
S = sum(U(indx,:).*V(indy,:),2);
UV = sparse(indx,indy,S,m,n);
tstart = tic;
iter_conv = 0;

% Main body of the algorithm ---------------------------------------------
 for k=1:Max_Iter
         
           UVold =  UV_true;
      %Column pruning mechanism ------------------------------------------
         temp = sum(U.^2,1);
         I = temp <= threshold; %1e-1 : threshold for pruning column based on its power
        % J = temp > 1e-1;
         U(:,I) = [];
         V(:,I) = [];
         D(I,:) = [];
         D(:,I) = [] ;
       %%-----------------------------------------------------------------
       
          U = update_U(U,V,D,Ys,indx,indy,m,n); 
         [D]  = update_D(U,V,lambda,p);
         [V,UV] = update_V(U,V,D,Ys,indx,indy,m,n);  
         [D,d]  = update_D(U,V,lambda,p);
         telapsed(k) = toc(tstart); 
   
         UV_true = U*V';
        %Estimate performance metrics and stopping criterion
         relative_error(k) = norm(UVold - UV_true,'fro')/norm(UVold,'fro');
        
         
         %use the following 2 lines if Y_true is a sparse matrix containing the test
         %set
         %S = sum(U(indx_true,:).*V(indy_true,:),2);
        % UV_true = sparse(indx_true,indy_true,S(:));
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %NMAE(k) = sum(sum(abs(Ytrue - UV_true)))/(4*9430); %NMAE for movielens 100k
         NRE(k) = norm(Ytrue - UV_true,'fro')/norm(Ytrue,'fro');
         cost_ = cost(Y,UV,d,lambda,p);
         cost_iter(k+1) = cost_;
         rel_obj(k) =  (cost_iter(k) - cost_ )/cost_iter(k);
     
         %cost_iter(k)
         %Check value of the stopping criterion
          
         if relative_error(k) < tol 
             
        %-----------------------------------------------
        %Rank one updates routine
        
        
            [u,sigma,v] = svds(Y - UV,1);
            mu = ((2-2*p)/(2-p))*sigma;
            
            
         %Check condition
            if rank_upd == 1 && lambda - (mu^(1-p))*sigma + .5*mu^(2-p) <=0 
                disp('rank one update');
                t_opt = sqrt(mu);
                u_ = t_opt*u;
                v_ = t_opt*v;
                U = [U u_];
                V = [V v_];
                d_last = (1/2^p)*lambda*p*(u_'*u_ + v_'*v_+ eps)^(p-1);
                d = diag(D);
                d_ = [d' d_last];
                D = diag(d_);
            else
              disp(['VarSchatten-' num2str(p) '  converged at iteration=' num2str(k) '/' num2str(Max_Iter)  'estimated rank=' num2str(rank(U*V'))]);
              break;
            end
         end
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 end
   toc 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function U = update_U(U,V,D,Ys,indx,indy,m,n)
   Y = sparse(indx,indy,Ys,m,n);
   S = sum(U(indx,:).*V(indy,:),2);
   UV = sparse(indx,indy,S,m,n);
   U = U - ((V'*V + D)\(((UV-Y)*V + U*D))')';
end

function [V,UV] = update_V(U,V,D,Ys,indx,indy,m,n)
   Y = sparse(indx,indy,Ys,m,n);
   S = sum(U(indx,:).*V(indy,:),2);
   UV = sparse(indx,indy,S,m,n);
   V = V - ((U'*U + D)\((UV-Y)'*U + V*D)')';
end

function [D,d,l2] = update_D(U,V,lambda,p)
        l2 = size(U,2);  
         for l=1:l2
           d(l) = ((U(:,l)'*U(:,l) + V(:,l)'*V(:,l) )^(1-p));
         end
        D = diag((1/2^p)*lambda.*p*(d.^(-1)));
end

function obj = cost(Y,UV,d,lambda,p)
         obj =   .5*norm(Y-UV,'fro')^2 + (1/2^p)*lambda*sum(d.^(2*p-1)) ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
