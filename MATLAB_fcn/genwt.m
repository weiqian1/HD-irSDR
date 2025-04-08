%%% generate the weights for penalty 

function vt = genwt(gamma, gg, einf)
    
    p = size(gamma,1);
    vt = zeros(p,1);
    minv = 1e20;
    for k = 1:p
        vnorm = norm(gamma(k,:));
        if vnorm < einf
            vt(k) = Inf;
        else 
            vt(k) = 1/vnorm^gg;
            if vt(k) < minv
                minv = vt(k);
            end
        end      
    end
    vt = vt./minv;
end