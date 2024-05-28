function RI = RandIndex(class_id,class_id_ture)
%RANDINDEX class_id,class_id_ture are two vectors representing the label of the dataset.
TP = 0;
TN = 0;
N = length(class_id_ture);
C_N2 = nchoosek(N,2);
for i = 1:N
    for j = (i+1):N
        if (class_id(i)==class_id(j) && class_id_ture(i)==class_id_ture(j));
        TP = TP+1;
        end
        if (class_id(i)~=class_id(j) && class_id_ture(i)~=class_id_ture(j));
        TN = TN+1;
        end
    end
end
RI = (TP+TN)/C_N2;  
