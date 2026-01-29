function Y = mlp_K_Pred(X, model)
% Multilayer perceptron prediction para dados no formato N x D
% Input:
%   model: model structure. Assumes model.b are ROW vectors.
%   X: n x d data matrix (n amostras, d features)
% Ouput:
%   Y: n x p response matrix (n amostras, p saídas)
% Adaptado de Mo Chen (sth4nth@gmail.com).

W = model.W;
b = model.b;
T = length(W);


% Itera pelas camadas ocultas
for t = 1:T-1
    % A ORDEM DA MULTIPLICAÇÃO MUDA E O TRANSPOSE (') É REMOVIDO
    % Y*W{t}  => (n x d_in) * (d_in x d_out) = (n x d_out)
    % b{t} deve ser um vetor linha 1 x d_out para ser adicionado a cada amostra
    X = tanh(X*W{t} + b{t});
end

% Camada de saída (mesma alteração)
Y = X*W{T} + b{T};
