%%% compute SCAD penalty derivatives
function gscad = pscad(gamma,lambda,a)
    if (isempty(a))
        a = 3.7;
    end
    p = size(gamma,1);
    gscad = zeros(p,1);
    for k = 1:p
        t = norm(gamma(k,:));
        if t <= lambda
            gscad(k) = 1;
        else
            gscad(k) = max(a-t/lambda,0)/(a-1);
        end        
    end
end