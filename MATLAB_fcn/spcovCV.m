%%%% sparse covariance esitmation by thresholding with CV
function sparsecov = spcovCV(X, lamseq, standard, method, nsplits, ntr)
    [n,~] = size(X);
    sampCov = cov(X)*(n-1)/n;
    if (isempty(ntr))
       ntr = floor(n*(1-1/log(n)));
    end
    nval = n-ntr;
    
    if (isempty(lamseq) && standard)
        lamseq = 0:.05:(1-.05);
    elseif (isempty(lamseq) && (~standard))
        absCov = abs(sampCov);
        maxval = max(max(absCov-diag(diag(absCov))));
        lamseq = linspace(0,maxval,20);
    end
    
    cvloss = zeros(size(lamseq,2),nsplits);
    for ns = 1:nsplits
        ind = randsample(n,n);
        indtr = ind(1:ntr);
        indval = ind((ntr+1):n);
        sstr = cov(X(indtr,:))*(ntr-1)/ntr;
        ssva = cov(X(indval,:))*(nval-1)/nval;
        
        for i = 1:size(lamseq,2)
            outCov = spcov(sstr, lamseq(i), standard, method);
            cvloss(i,ns) = norm(outCov-ssva,'fro')^2;
        end        
    end
    
    cverr = sum(cvloss,2);
    [~,cvix] = min(cverr);
    bestlam = lamseq(cvix);
    
    outCov1 = spcov(sampCov, bestlam, standard, method);
    
    % output 
    sparsecov.sigma = outCov1;
    sparsecov.bestlam = bestlam;
    sparsecov.cverr = cverr;
    sparsecov.lamseq = lamseq;
    sparsecov.ntr = ntr;
    
end