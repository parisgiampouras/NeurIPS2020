function [U,V] = VarSchatten(Y,lambda,L,Ytrue,Max_Iter,tol,threshold,p,rank_upd)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%
%Code for VarSchatten algorithm associated with the paper:
% P. Giampouras, R. Vidal, A. Rontogiannis, B. Haeffele, "A novel
% variational form of the Schatten-p quasi-norm", Neurips 2020.
% Written by Paris Giampouras, 10/2020 e-mail: parisg@jhu.edu
%
%
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


%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 9
    rank_upd = 1;
end
if nargin < 8
    p = 0.5;
end
if nargin < 7
    threshold = 1e-5;
end
if nargin < 6
    tol = 1e-5;
end
if nargin < 5
    Max_Iter= 500;
end




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

Y = sparse(indx,indy,Ys,m,n);
S = sum(U(indx,:).*V(indy,:),2);
UV = sparse(indx,indy,S,m,n);

tstart = tic;

% Main body of the algorithm ---------------------------------------------
 for k=1:Max_Iter
         
           UVprev =  U*V';
      %%Column pruning mechanism ------------------------------------------
           I = sum(U.^2,1) <= threshold; % threshold for pruning column based on its power
        % J = temp > 1e-1;
           U(:,I) = [];
           V(:,I) = [];
           D(I,:) = [];
           D(:,I) = [] ;
       %%-----------------------------------------------------------------
          
          U = update_U(U,V,D,Ys,indx,indy,m,n); 
          
          if size(U,2) == 0
              [U,V] = rank_one_update(Y,UV,U,V,D,lambda,p);
          end
          
         [D]  = update_D(U,V,lambda,p);
         [V,UV] = update_V(U,V,D,Ys,indx,indy,m,n);  
         [D,d]  = update_D(U,V,lambda,p);
         telapsed(k) = toc(tstart); 
   
         UV_curr = U*V';
        %Estimate performance metrics and stopping criterion
         relative_error(k) = norm(UV_curr-UVprev,'fro')/norm(UVprev,'fro');

         %Check value of the stopping criterion
          
         if relative_error(k) < tol 
             
        %-----------------------------------------------
        %Rank one updates routine
        
        
            [rank_upd_activated] = check_condition_rank_one(Y,UV,lambda,p);
            
         %Check condition
            if rank_upd == 1 && rank_upd_activated 
                disp('rank one update');
                [U,V,D] = rank_one_update(Y,UV,U,V,D,lambda,p);
               
            else
              disp(['-----VarSchatten-' num2str(p) '  converged at iteration= ' num2str(k) '/' num2str(Max_Iter)  ',  estimated rank=' num2str(rank(U*V')) '--------']);
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


function [U,V,D] = rank_one_update(Y,UV,U,V,D,lambda,p)
            [u,sigma,v] = svds(Y - UV,1);
            mu = ((2-2*p)/(2-p))*sigma;
             t_opt = sqrt(mu);
                u_ = t_opt*u;
                v_ = t_opt*v;
                U = [U u_];
                V = [V v_];
                d_last = (1/2^p)*lambda*p*(u_'*u_ + v_'*v_+ eps)^(p-1);
                d = diag(D);
                d_ = [d' d_last];
                D = diag(d_);
            
end

function [rank_upd_activated] = check_condition_rank_one(Y,UV,lambda,p)
        [u,sigma,v] = svds(Y - UV,1);
         mu = ((2-2*p)/(2-p))*sigma;
       rank_upd_activated =  lambda - (mu^(1-p))*sigma + .5*mu^(2-p) <=0;
end