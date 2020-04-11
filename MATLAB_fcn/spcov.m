function outCov = spcov(s, lam, standard, method)
        
    if (standard)
        dhat = diag(sqrt(diag(s)));
        dhatinv = diag(1./sqrt(diag(s)));
        S = dhatinv*s*dhatinv;
        
        if (strcmp(method,'soft'))
            tmp = abs(S)-lam;
            tmp = tmp.*(tmp>0);
            Ss = sign(S).*tmp;
            Ss = Ss - diag(diag(Ss));
            Ss = Ss + diag(diag(S));
        end 
        outCov = dhat*Ss*dhat;        
    else
        S = s;
        if (strcmp(method,'soft'))
            tmp = abs(S)-lam;
            tmp = tmp.*(tmp>0);
            Ss = sign(S)*tmp;
            Ss = Ss - diag(diag(Ss));
            Ss = Ss + diag(diag(s));
        end
        outCov = Ss;
    end
end