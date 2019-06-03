clc;
clear;
digits(5);
rng(100);
n = 100;
C_list = [0.1 100 10 100];
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);
Xi = [class1;class2];
yi = [ones(60,1);-1*ones(40,1)];
%%
for j = 1:2
    C = C_list(j);
    figure(j);
    scatter(Xi(yi == 1,1),Xi(yi==1,2),80,'r','s','filled','DisplayName','Class 1','LineWidth',1.5,'MarkerEdgeColor',[0.5 0.5 0.5]);
    hold on;
    scatter(Xi(yi == -1,1),Xi(yi==-1,2),50,'g','o','filled','DisplayName','Class 2','LineWidth',1.5,'MarkerEdgeColor',[0.5 0.5 0.5]);
    
    xlabel('X1');
    ylabel('X2');
    G = (yi*yi') .*( Xi*Xi');
    e = ones (n ,1);
    % l = eps ^.5;
    % G = G + l*eye(n);    
    ad = quadprog(G,-e,[],[],yi',0,zeros(n,1),C*ones(n,1));
    ad = round(ad,5);
    wqp = [0 0];
    for i = 1:100
        wqp = wqp + ad(i)*yi(i)*Xi(i,:);
        disp(wqp);
    end
    wqp = wqp';
    b = 0.3126; % For C = 0.1
    if ( C == 100 )
        b = 0.1958;
    end 
    b0 = -b/wqp(2);
    b1 = -wqp(1)/wqp(2);
    f = @(x) b0+b1*x;
    fplot( f, [-3, 10],'black','DisplayName','Decision Boundary' )

    hold on
    b0 = (-b+1)/wqp(2);
    b1 = -wqp(1)/wqp(2);
    f = @(x) b0+b1*x;
    fplot( f, [-3, 10],'--black','DisplayName','Margin' )


    b0 = (-b-1)/wqp(2);
    b1 = -wqp(1)/wqp(2);
    f = @(x) b0+b1*x;
    fplot( f, [-3, 10],'--black','DisplayName','Margin' )

    hold on

    res  = yi.*[Xi*wqp+b];
    res = round(res,5);
    
    title(['Linear SVM C = ',num2str(C),' Support Vectors = ',num2str(sum(ad>0)),' Misclassifications = ',num2str(length(Xi(res < 0,:)))]);
    scatter(Xi(( ad == C) & (ad > 0),1),Xi((ad == C) & (ad > 0),2),60,'x','b','LineWidth',1.5,'DisplayName','Inside or Misclassified');
    hold on
    scatter(Xi(( ad < C) & (ad > 0),1),Xi((ad < C) & (ad > 0),2),60,'.','b','LineWidth',2,'DisplayName','Margin');
    hold on;
    scatter(Xi(res < 0,1),Xi(res < 0,2),100,'d','DisplayName','Misclassifications');
    xlim([-2 8])
    ylim([-2 6])
    legend
    hold off
end
%%
for m=3:4
    C = C_list(m);
    nsq=sum(Xi.^2,2);
    K=bsxfun(@minus,nsq,(2*Xi)*Xi.');
    K=bsxfun(@plus,nsq.',K);
    K=exp(-K/(2*1.75^2));

    G = (yi*yi') .* K;
    e = ones (n ,1);
    % l = eps ^.5;
    % G = G + l*eye(n);
    tic
    ad = quadprog(G,-e,[],[],yi',0,zeros(n,1),C*ones(n,1));
    ad = round(ad,3);

    d = weight(Xi,yi,ad);
    results = [];
    sup_vecs = find(ad>0 & ad<C);
    final = 0;
    syms x1 x2;
    for i=1:100
        s = d(x1,x2);    
        x1 = Xi(i,1);
        x2 = Xi(i,2);
        o = vpa(subs(s));
        results(end+1) = o;
    end
    for j = 1:length(sup_vecs)
        final = final + (yi(sup_vecs(j)) - results(sup_vecs(j)));
    end
    b = final/length(sup_vecs);
    % b = -2.6129;
    % if (C == 10)
    %     b = -0.4802;
%     % end
%     if (C==10)
%         b = + 0.1;
%     else
%         b = + 0.07;
%     end

    results = results + b;



    xx = -2:0.2:8;
    yy = -2:0.2:6;
    [X,Y] = meshgrid(xx,yy);
    sz = size(X);
    Z = [[]];
    for i = 1:sz(1)
        for j = 1:sz(2)
            s = d(x1,x2);    
            x1 = X(i,j);
            x2 = Y(i,j);
            o = vpa(subs(s));
            Z(i,j) = o + b;        
        end
    end

    % Need to subtract 0.07 for C = 100 and subtract 0.1 for C = 10. Both using
    % only margin vectors
    figure(m);
    contourf(X,Y,Z,[0 0],'LineStyle','-','Fill','off','DisplayName','Decision Boundary');
    hold on;
    contourf(X,Y,Z,[1 1],'LineStyle','--','Fill','off','DisplayName','Margin Boundary');
    hold on;
    contourf(X,Y,Z,[-1 -1],'LineStyle','--','Fill','off','DisplayName','Margin Boundary');
    hold on;
    scatter(Xi(yi == 1,1),Xi(yi==1,2),50,'r','o','filled','DisplayName','Class 1','LineWidth',1.5,'MarkerEdgeColor',[0.5 0.5 0.5]);
    hold on;
    scatter(Xi(yi == -1,1),Xi(yi==-1,2),80,'g','square','filled','DisplayName','Class 2','LineWidth',1.5,'MarkerEdgeColor',[0.5 0.5 0.5]);
    hold on;
    scatter(Xi(( ad < C) & (ad > 0),1),Xi((ad < C) & (ad > 0),2),60,'.','blue','LineWidth',2,'DisplayName','Margin');
    hold on
    scatter(Xi(( ad == C) & (ad > 0),1),Xi((ad == C) & (ad > 0),2),60,'x','blue','LineWidth',1.5,'DisplayName','Missclassification/Violation');
    hold on;
    results = yi'.*results;
    scatter(Xi(results < 0,1),Xi(results < 0,2),100,'d','DisplayName','Misclassifications');
    xlabel('X1');
    ylabel('X2');
    title(['Gaussian RBF kernel SVM C = ',num2str(C),' Support Vectors = ',num2str(sum(ad>0)),' Misclassifications = ',num2str(length(Xi(results < 0,:)))]);

    hold off
    xlim([-2 8])
    ylim([-2 6])
    legend    
end
%%
sizes = [];
for i = 1:5
    sizes(end+1) = i*100;
end
C = [10 100];

kc1_vals = [];
kc2_vals = [];
sc1_vals = [];
sc2_vals = [];
for n = 1:5
    class1=mvnrnd([1 3],[1 0; 0 1],.6*sizes(n));
    class2=mvnrnd([4 1],[2 0; 0 2],.4*sizes(n));
    Xi = [class1;class2];
    yi = [ones(.60*sizes(n),1);-1*ones(.40*sizes(n),1)];
    kc1 = 0;sc1 = 0;kc2 = 0;sc2 = 0;
    for j = 1:5
        tic
        nsq=sum(Xi.^2,2);
        K=bsxfun(@minus,nsq,(2*Xi)*Xi.');
        K=bsxfun(@plus,nsq.',K);
        K=exp(-K/(2*1.75^2));
        G = (yi*yi') .* K;
        e = ones (sizes(n) ,1);       
        ad = quadprog(G,-e,[],[],yi',0,zeros(sizes(n),1),C(1)*ones(sizes(n),1));
        ad = round(ad,3);
        kc1 = kc1 + toc;
        
        tic
        cl = fitcsvm(Xi,yi,'KernelFunction','rbf',...
    'KernelScale',1/sqrt((1/(2*1.75^2))),'BoxConstraint',C(1),'ClassNames',[-1,1]);
        sc1 = sc1 + toc;
        
        tic
        nsq=sum(Xi.^2,2);
        K=bsxfun(@minus,nsq,(2*Xi)*Xi.');
        K=bsxfun(@plus,nsq.',K);
        K=exp(-K/(2*1.75^2));
        G = (yi*yi') .* K;
        e = ones (sizes(n) ,1);       
        ad = quadprog(G,-e,[],[],yi',0,zeros(sizes(n),1),C(2)*ones(sizes(n),1));
        ad = round(ad,3);
        kc2 = kc2 + toc;
        
        tic
        cl = fitcsvm(Xi,yi,'KernelFunction','rbf',...
    'KernelScale',(2*(1.75)^2),'BoxConstraint',C(2),'ClassNames',[-1,1]);
        sc2 = sc2 + toc;
    end
    kc1 = kc1/5;
    kc2 = kc2/5;
    sc1 = sc1/5;
    sc2 = sc2/5;
    
    kc1_vals(end+1) = kc1;
    kc2_vals(end+1) = kc2;
    sc1_vals(end+1) = sc1;
    sc2_vals(end+1) = sc2;  
end

figure(5)
plot(sizes,kc1_vals,'r','DisplayName','Quadprog C = 10');
hold on
plot(sizes,kc2_vals,'g','DisplayName','Quadprog C = 100');
hold on
plot(sizes,sc1_vals,'b','DisplayName','SMO C = 10');
hold on
plot(sizes,sc2_vals,'y','DisplayName','SMO C = 100');
hold on
title('Performance comparison of kernel SVM and SMO');
xlabel('Size of dataset');
ylabel('Time taken to complete computation (seconds)');
legend
%%
function d = weight(Xi,yi,ad)
    syms x1 x2;    
    x = [x1*ones(100,1) x2*ones(100,1)];
    X_m = Xi-x;
    sq = sum(X_m.^2,2);
    sq_by_sigma = sq./(2*(1.75^2));
    sq_e = exp(-sq_by_sigma);
    result = sq_e.*yi.*ad;
    final = sum(result,1);
    d = @(x1,x2) final;
end