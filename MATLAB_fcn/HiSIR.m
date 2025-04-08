function [gamma, V, k] = HiSIR(M, S, ns, G, lambda1, vt, gamma0, V0, maxit, eps,pen,a)
% solve integrative surrogate objective function

if (isempty(maxit))
    maxit = 2000;
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


[p,h] = size(M);

pv = zeros(1,ns); 
dv = zeros(1,ns); 
idx = cell(1,ns); 
gamma = gamma0;
V = V0;
ww1 = zeros(1,ns); 
Bm = zeros(p,h); 

for s = 1:ns
    [pt,dt] = size(gamma0{s});
    pv(s) = pt;
    dv(s) = dt;
    idx{s} = (G==s);
    ww1(s) = real(eigs(S(idx{s},idx{s}),1));
    Bm(idx{s},:) = gamma{s}*V{s};
end
    
% start loop
for k = 1:maxit    
    gamma0 = gamma;
    V0 = V;
    dif0 = zeros(1,ns);    
    for s = 1:ns
        tidx = idx{s};
        MS = M(tidx,:)-S(tidx,~tidx)*Bm(~tidx,:);
        gg = -MS*V{s}'+S(tidx,tidx)*gamma{s}; 
        tgamma = gamma{s};
        tlambda1 = lambda1(s);
        tvt = vt(tidx);
        if strcmp(pen,'scad')
            alpha = pscad(tgamma,tlambda1,a);
            tvt = tvt(:).*alpha(:);
        end        
        ww = ww1(s);
        td = dv(s);        
        for j = 1:pv(s)
            oldgamj = tgamma(j,:)';
            vzj = -gg(j,:)'+ww*oldgamj;
            tn = 1-tlambda1*tvt(j)/norm(vzj);
            if tn > 0
                newgamj = tn/ww*vzj;
            else
                newgamj = zeros(td,1);
            end
            tgamma(j,:) = newgamj;
            difgamma = newgamj-oldgamj;
            if (norm(difgamma)>100)
                tgamma(j,:) = zeros(td,1);
            end            
        end
        gamma{s} = tgamma;
        zg = MS'*tgamma;
        [W1,~,W2] = svd(zg,0);
        V{s} = W2*W1';
        Bm(tidx,:) = tgamma*V{s};
        
        difgamma = (tgamma-gamma0{s}).^2;
        difV = (V{s}-V0{s}).^2;
        dif0(s) = max([difgamma(:);difV(:)]);        
    end
    
    dif = max(dif0);
    if dif < eps
        break
    end
        
end
end
