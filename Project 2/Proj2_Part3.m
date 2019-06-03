clc;
clear;
sizes = [];
for i = 1:5
    sizes(end+1) = i*100;
end
C = [10 100];
%%
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
    for j = 1:3
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
    'KernelScale',(2*(1.75)^2),'BoxConstraint',C(1),'ClassNames',[-1,1]);
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
    kc1 = kc1/3;
    kc2 = kc2/3;
    sc1 = sc1/3;
    sc2 = sc2/3;
    
    kc1_vals(end+1) = kc1;
    kc2_vals(end+1) = kc2;
    sc1_vals(end+1) = sc1;
    sc2_vals(end+1) = sc2;
%     scatter(sizes(n),kc1,'b');
%     hold on
%     scatter(sizes(n),kc2,'r');
%     hold on
%     scatter(sizes(n),sc1,'g');
%     hold on
%     scatter(sizes(n),sc2,'y');
%     hold on    
end
%%
plot(sizes,kc1_vals,'r','DisplayName','Quadprog C = 10');
hold on
plot(sizes,kc2_vals,'g','DisplayName','Quadprog C = 100');
hold on
plot(sizes,sc1_vals,'b','DisplayName','SMO C = 10');
hold on
plot(sizes,sc2_vals,'y','DisplayName','SMO C = 100');
hold on
legend;