function mmtc_testrun

f = format('shorte');
m = 10;
k = 14;
n = 13;
A = ones(m,k,'single','gpuArray');
B = ones(k,n,'single','gpuArray');

trueC = double(A*(2^7-1))*double(B*(2^7-1));
C = mmtc(double(A)*(2^7-1),double(B)*(2^7-1),1);
max(abs((C-trueC)./trueC),[],'all')
trueC = double(A*(2^11-1))*double(B*(2^11-1));
C = mmtc(double(A)*(2^11-1),double(B)*(2^11-1),1);
max(abs((C-trueC)./trueC),[],'all')
C = mmtc(double(A)*(2^11-1),double(B)*(2^11-1),2);
max(abs((C-trueC)./trueC),[],'all')
C = mmtc(double(A)*(2^11-1),double(B)*(2^11-1),3);
max(abs((C-trueC)./trueC),[],'all')
C = double(single(A*(2^11-1))*single(B*(2^11-1)));
max(abs((C-trueC)./trueC),[],'all')

format(f);
end