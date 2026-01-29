
function model = mlp_K_Treino(X, Y_true, arq_red, hiper_par,s = "")
% Treina uma MLP para regressão usando gradiente descendente com mini-batch.
%   X: n x d matriz de dados de entrada (n amostras, d features)
%   Y_true: n x p matriz de dados de saída verdadeiros (n amostras, p saídas)
%   options: Estrutura com hiperparâmetros.
%       .arq_red: Vetor com o tamanho das camadas ocultas. Ex: [10 5] para 2 camadas
%                 ocultas com 10 e 5 neurônios, respectivamente.
%       .par_sim:
%               epocas:  Numero de épocas de treinamento
%               tx_ap: Taxa de aprendizagem
%               tam_batch:    Tamanho do mini-batch (ex:1, 32 ou 64)

% --- 1. Inicialização
  [n, d] = size(X);
  p = size(Y_true, 2);

  epocas = hiper_par.epocas;
  tx_ap = hiper_par.tx_ap;
  tam_batch = hiper_par.tam_batch;

  layer_sizes = arq_red;

  % Arquitetura e inicialização de pesos e vieses
  arch = [d, layer_sizes, p];
  T = length(arch) - 1;
  W = cell(1, T);
  b = cell(1, T);
  for t = 1:T
      limit = sqrt(6 / (arch(t) + arch(t+1)));
      W{t} = rand(arch(t), arch(t+1)) * 2 * limit - limit;
      b{t} = zeros(1, arch(t+1));
  end

  fprintf('Iniciando o treinamento por %d épocas com tam_batch=%d...\n', epocas, tam_batch);

  % --- 2. Loop de Treinamento com Mini-Batches ---
  for epoch = 1:epocas

      % MUDANÇA 1: Embaralhar os dados a cada época. Isso é CRUCIAL!
      % Se não embaralharmos, a rede verá os mesmos mini-batches na mesma ordem
      % e pode ficar presa em ciclos.
      idx = randperm(n);
      X_shuffled = X(idx, :);
      Y_true_shuffled = Y_true(idx, :);

      % MUDANÇA 2: Loop interno para iterar sobre os mini-batches
      for i = 1:tam_batch:n
          % Define os índices para o mini-batch atual
          end_idx = min(i + tam_batch - 1, n);

          % Cria o mini-batch
          X_batch = X_shuffled(i:end_idx, :);
          Y_batch_true = Y_true_shuffled(i:end_idx, :);
          current_batch_size = size(X_batch, 1); % O último batch pode ser menor

          % O resto do código (Forward, Backward, Update) agora fica DENTRO deste loop
          % e opera sobre o _batch_ e não sobre X e Y_true completos.

          % == PASSO A: FORWARD PROPAGATION (no batch) ==
          A = cell(1, T + 1);
          A{1} = X_batch;
          for t = 1:T-1
              A{t+1} = tanh(A{t} * W{t} + b{t});
          end
          Y_pred_batch = A{T} * W{T} + b{T};
          A{T+1} = Y_pred_batch;

          % == PASSO B: BACKWARD PROPAGATION (no batch) ==
          dW = cell(1, T);
          db = cell(1, T);

          % MUDANÇA 3: Normaliza o gradiente pelo tamanho do batch, não do dataset
          delta = (2/current_batch_size) * (Y_pred_batch - Y_batch_true);

          dW{T} = A{T}' * delta;
          db{T} = sum(delta, 1);

          for t = T-1:-1:1
              tanh_grad = 1 - A{t+1}.^2;
              delta = (delta * W{t+1}') .* tanh_grad;
              dW{t} = A{t}' * delta;
              db{t} = sum(delta, 1);
          end

          % == PASSO C: ATUALIZAÇÃO DOS PESOS E VIESES (acontece a cada mini-batch!) ==
          for t = 1:T
              W{t} = W{t} - tx_ap * dW{t};
              b{t} = b{t} - tx_ap * db{t};
          end
      end % Fim do loop de mini-batches

      % Monitoramento do Erro (calculado no dataset inteiro, uma vez por época, para ter uma visão estável)
   #   if mod(epoch, 100) == 0 || epoch == 1
          % Para calcular o erro total, precisamos fazer um forward pass completo
          model_t.W = W;
          model_t.b = b;
          Y_pred_full = mlp_K_Pred(X, model_t);
          loss(epoch) = mean(sum((Y_pred_full - Y_true).^2, 2));
      if strcmp(s, "show")
          figure(1)
          clf
          plot(Y_pred_full,'-',Y_true)
          hold on
          pause(0.0001)
          hold off
      end
          fprintf('Época %d/%d, Erro (MSE): %f\r', epoch, epocas, loss(epoch));
  #    end

  end % Fim do loop de épocas

  % Retorna o modelo treinado
  model.W = W;
  model.b = b;
  model.loss = loss;
  fprintf('\nTreinamento concluído.\n');
end


