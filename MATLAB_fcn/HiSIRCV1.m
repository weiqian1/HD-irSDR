%%% use CV to find best lambda and reduction dimension to solve HD-irSDR


function cvres = HiSIRCV1(X, yslice, ns, G, lambda_cand0, dseq, Kfold, vt, maxit, eps, pen,a)
% X: predictor
% yslice: sliced response
% ns: number of sources
% G: source index
% lambda_cand0: candidates for regularization parameter of each source
% dseq: candidates for reduction dimension
% Kfold: number of CV folds
% vt: weights for regularization parameters
% maxit: maximum number of iterations
% eps: tolerance
% pen: choose a penalty option, 'scad' or 'lasso'
% a: parameter used only for 'scad' penalty

[n,~] = size(X);
h = max(yslice);
pv = zeros(ns,1);
for s = 1:ns
    pv(s) = sum(G==s);
end

if (isempty(maxit))
    maxit = 1000;
end

if (isempty(eps))
    eps = 1e-12;
end

if (isempty(pen))
    pen = 'scad';
end

if (isempty(a))
    a = 3.7;
end
method = 'sir';

% make the splits
ind = randsample(n,n);
flen = floor(n/Kfold);
fstart = zeros(1,Kfold);
fend = zeros(1,Kfold);
for k = 1:(Kfold-1)
    fstart(k) = (k-1)*flen+1;
    fend(k) = k*flen;
end
fstart(Kfold) = (Kfold-1)*flen+1;
fend(Kfold) = n;

% start CV 
[nss0,lamb_num] = size(lambda_cand0);
[dlen,nss] = size(dseq);
if (nss~=ns || nss0~=ns)
    error("Numbers of sources do not match for ns, dseq and lambda_cand0")
end

cvloss0 = zeros(lamb_num*dlen,Kfold);

parfor k = 1:Kfold
    indval = ind(fstart(k):fend(k)); %#ok<PFBNS>
    indtr = setdiff(1:n,indval);
    
    Xtr = X(indtr,:); %#ok<PFBNS>
    Xval = X(indval,:);
    Ytr = yslice(indtr); %#ok<PFBNS>
    Yval = yslice(indval);
    
    SDRker = sdrkernelt(Xtr,Ytr,h,method,h,[],[]);
    M = SDRker.Xybar;
    S = SDRker.hatSigma;
    
    cvlossK = Inf(lamb_num*dlen,1);
    for dix = 1:dlen
        d = dseq(dix,:); %#ok<PFBNS>
        fprintf('running CV fold = %d, ',k); fprintf('d = '); fprintf('%d ',d);fprintf('.\n');
        gamma0 = cell(1,ns);
        V0 = cell(1,ns);
        pp = 5;
        for s = 1:ns
            gamma0{s} = [zeros(pp-d(s),d(s));eye(d(s));zeros(pp-d(s),d(s));eye(d(s));zeros(pv(s)-2*pp,d(s))]/10; %#ok<PFBNS>
            V0{s} = (ones(d(s),h))/sqrt(h)/10;
        end
        [gamma,V,~] = HiSIR(M,S,ns,G,lambda_cand0(:,1),vt,gamma0,V0,maxit,eps,pen,a); %#ok<PFBNS>
        cvlossK((dix-1)*dlen+1) = Inf;
        for i = 2:lamb_num
            [gamma,V,~] = HiSIR(M,S,ns,G,lambda_cand0(:,i),vt,gamma,V,maxit,eps,pen,a);
            Xtrain = [];
            Xtest = [];
            flg = false;
            for s = 1:ns
                gamma00 = gamma{s};
                p0 = sum(arrayfun(@(ix) norm(gamma00(ix,:)), 1:pv(s))>1e-5); 
                if p0 < d(s)
                    flg = true;
                    break
                end
                if cond(gamma00'*gamma00) > 1000
                    flg = true;
                    break
                end
                gammatemp = gamma00/(gamma00'*gamma00);
                Xtrain = [Xtrain,Xtr(:,G==s)*gammatemp];
                Xtest = [Xtest,Xval(:,G==s)*gammatemp];               
            end
            if (flg==true)
                cvlossK((dix-1)*dlen+1) = Inf;
            else
                temptree = fitctree(Xtrain,Ytr);
                m = max(temptree.PruneList)-1;
                [~,~,~,bestLevel] = cvloss(temptree,'SubTrees',0:m,'KFold',10);
                temptree = prune(temptree,'Level',bestLevel);
                label = predict(temptree,Xtest);
                cvlossK((dix-1)*lamb_num+i) = sum(label~=Yval);
            end                        
        end       
    end
    cvloss0(:,k) = cvlossK;
end

% find the best d and lambda
cverr = sum(cvloss0,2);
cverr = reshape(cverr,[lamb_num,dlen]);
for i = 1:dlen
    cverr(:,i) = movingmean(cverr(:,i),5);
end
cverr = cverr(:);
[~,cvix] = min(cverr);
dix = floor((cvix-1)/lamb_num)+1;
d = dseq(dix,:);
lamix = mod((cvix-1),lamb_num)+1;
bestlam = lambda_cand0(:,lamix);

fprintf('d = ');
fprintf('%d ',d);
fprintf(', lambda = ');
fprintf('%d ',bestlam);
fprintf('.\n');

% fit the path using best d to find the best gamma
SDRker = sdrkernelt(X,yslice,h,method,h,[],[]);
M = SDRker.Xybar;
S = SDRker.hatSigma;

gamma_res = cell(1,ns);
V_res = cell(1,ns);
k_res = zeros(lamb_num);
for s = 1:ns
    gamma_res{s} = zeros(pv(s),d(s),lamb_num);
    V_res{s} = zeros(d(s),h,lamb_num);
end

% initial values
gamma0 = cell(1,ns);
V0 = cell(1,ns);
pp = 4;
for s = 1:ns
    ps = pv(s);
    if (ps < 10)
        error('p for each source should be >=10');
    end
    gamma0{s} = [zeros(pp-d(s),d(s));eye(d(s));zeros(pp-d(s),d(s));eye(d(s));zeros(pv(s)-2*pp,d(s))]/10;
    V0{s} = (ones(d(s),h))/sqrt(h)/10;
end

[gamma,V,k] = HiSIR(M, S, ns, G, lambda_cand0(:,1), vt, gamma0, V0, maxit, eps,pen,a);
for s = 1:ns
    gamma_res{s}(:,:,1) = gamma{s};
    V_res{s}(:,:,1) = V{s};
end
k_res(1) = k;

for i = 2:lamix
    if mod(i,10)==0
        fprintf('running %d th lambda.\n',i);
    end
    [gamma,V,k] = HiSIR(M, S, ns, G, lambda_cand0(:,i), vt, gamma, V, maxit, eps,pen,a);
    for s = 1:ns
        gamma_res{s}(:,:,i) = gamma{s};
        V_res{s}(:,:,i) = V{s};
    end
    k_res(i) = k;
end
bestgam = gamma;

% summary
cvres.bestd = d;
cvres.bestlam = bestlam;
cvres.bestgam = bestgam;
cvres.bestV = V;
cvres.lamix = lamix;
cvres.dseq = dseq;
cvres.lamb = lambda_cand0;
cvres.Kfold = Kfold;
cvres.cverr = reshape(cverr,[lamb_num,dlen]);
cvres.gamma = gamma_res;
cvres.V = V_res;
cvres.k = k_res;
 

end
