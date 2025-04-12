%%%%% This is an example to run HD-irSDR for a sparse semiparametric model
%%%%% Case 2

%%% simulate a high-dimensional data
rng(99999); 
n = 200;
p1 = 500;
p2 = 500;
p = p1+p2;
ns = 2;
G = [ones(1,p1),2*ones(1,p2)];

Sigma = eye(p);
rho = 0.5;
for i = 1:(p-1)
    for j = (i+1):p
        Sigma(i,j) = rho^(j-i);
        Sigma(j,i) = Sigma(i,j);
    end
end
mu = zeros(p,1);
X = mvnrnd(mu,Sigma,n);
sigma = 1;
epsilon = normrnd(0,sigma,n,1);
q1 = 3;
q21 = 2;
q22 = 2;
Gam1 = [zeros(p1-q1,1);ones(q1,1)];
Gam21 = [[1;-1];zeros(p2-q21,1)];
Gam22 = [zeros(p2-q22,1);ones(q22,1)];
Gam2 = [Gam21,Gam22];
y = 0.25*X(:,1:p1)*Gam1 + sign(2*X(:,(p1+1):p)*Gam21).*log(5*abs(X(:,(p1+1):p)*Gam22+5))+0.2*epsilon;

% making slices and set candidate pamameters
h = 5;
prob = linspace(0,1,h+1);
edges = quantile(y,prob);
yslice = discretize(y,edges);
lamb_max = 1;
lamb_min = .05;
lamb_num = 50;
lambda_cand = exp(log(lamb_max)-(0:(lamb_num-1))*log(lamb_max/lamb_min)/(lamb_num-1));
lambda_cand0 = repmat(lambda_cand,[ns,1]);
Kfold = 10;
dseq = [1,1;1,2];

%%%% run HD-irSDR
vt = ones(p,1);
a = 10;
cvres = HiSIRCV1(X, yslice, ns, G, lambda_cand0, dseq, Kfold, vt, [], [],[], a);
gamma_best = cvres.bestgam;
V_best = cvres.bestV;
bestlam = cvres.bestlam;
lamix = cvres.lamix;
gamma_est1 = gamma_best{1};
gamma_est2 = gamma_best{2};




