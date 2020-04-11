%%% construct SDR kernel matrix
function kernelres = sdrkernelt(X, yslice, h, method, t, lambda0, ~)
% lambda0: thresholding parameter for Sigma
    [n,p] = size(X);
    if (strcmp(method,'pfc'))
        Xybar = (X'*yslice)/(yslice'*yslice);
        epshat = (eye(n)-yslice/(yslice'*yslice)*yslice')*X;
        hatSigma = cov(epshat)*(n-1)/n;
    elseif (strcmp(method,'spfc'))
        Xybar = (X'*yslice)/(yslice'*yslice);
        %Xybar = (X'*yslice);
        epshat = (eye(n)-yslice/(yslice'*yslice)*yslice')*X;
        temps = cov(epshat)*(n-1)/n;
        hatSigma = spcov(temps, lambda0, true, 'soft');
    elseif (strcmp(method,'sir'))
        xbar = mean(X);
        Xybar = zeros(p,h);
        for k = 1:h
            index = (yslice==k);
            if (sum(index)~=0)
                Xybar(:,k) = (mean(X(index,:))'-xbar')*sqrt(sum(index)/n);
            else
                Xybar(:,k) = zeros(p,1);
            end
        end
        Xybar = Xybar(:,1:t);
        hatSigma = cov(X)*(n-1)/n;
    elseif (strcmp(method,'ssir'))
        xbar = mean(X);
        Xybar = zeros(p,h);
        for k = 1:h
            index = (yslice==k);
            if (sum(index)~=0)
                Xybar(:,k) = (mean(X(index,:))'-xbar')*sqrt(sum(index)/n);
            else
                Xybar(:,k) = zeros(p,1);
            end
        end
        Xybar = Xybar(:,1:t);
        temps = cov(X)*(n-1)/n;
        hatSigma = spcov(temps, lambda0, true, 'soft');
    end 
    kernelres.Xybar = Xybar;
    kernelres.hatSigma = hatSigma;   
end