#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <string>

struct ANNImpl : public torch::nn::Module {
 public:
  ANNImpl(const int32_t num_hidden = 128)
      : fc1_(28 * 28, num_hidden), fc2_(num_hidden, 10) {
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
  }

  torch::Tensor forward(const torch::Tensor& in) {
    torch::Tensor x = in;
    x = torch::flatten(x, 1);
    x = fc1_->forward(x);
    x = torch::relu(x);
    x = fc2_->forward(x);
    x = torch::log_softmax(x, 1);

    return x;
  }

 private:
  torch::nn::Linear fc1_;
  torch::nn::Linear fc2_;
};

TORCH_MODULE(ANN);

struct DNNImpl : public torch::nn::Module {
 public:
  DNNImpl(const int32_t num_hidden = 128)
      : conv1_(torch::nn::Conv2dOptions(1, 32, 3)),
        conv2_(torch::nn::Conv2dOptions(32, 64, 3)),
        dropout1_(0.25),
        dropout2_(0.5),
        fc1_(9216, num_hidden),
        fc2_(num_hidden, 10) {
    register_module("conv1", conv1_);
    register_module("conv2", conv2_);
    register_module("dropout1", dropout1_);
    register_module("dropout2", dropout2_);
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
  }

  torch::Tensor forward(const torch::Tensor& in) {
    torch::Tensor x = in;

    x = conv1_->forward(x);
    x = torch::relu(x);
    x = conv2_->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 2);
    x = dropout1_->forward(x);
    x = torch::flatten(x, 1);
    x = fc1_->forward(x);
    x = torch::relu(x);
    x = dropout2_->forward(x);
    x = fc2_->forward(x);
    x = torch::log_softmax(x, 1);

    return x;
  }

 private:
  torch::nn::Conv2d conv1_;
  torch::nn::Conv2d conv2_;
  torch::nn::Dropout dropout1_;
  torch::nn::Dropout dropout2_;
  torch::nn::Linear fc1_;
  torch::nn::Linear fc2_;
};

TORCH_MODULE(DNN);

template <typename Model, typename DataLoader>
void train(Model& model, const torch::Device device, DataLoader& data_loader,
           torch::optim::Optimizer& optimizer) {
  model.train();

  for (auto& batch : data_loader) {
    optimizer.zero_grad();

    const torch::Tensor data{batch.data.to(device)};
    const torch::Tensor label{batch.target.to(device)};

    const torch::Tensor out{model.forward(data)};

    torch::Tensor loss{torch::nll_loss(out, label)};

    loss.backward();

    optimizer.step();
  }
}

template <typename Model, typename DataLoader>
void test(Model& model, const torch::Device device, DataLoader& data_loader) {
  torch::NoGradGuard no_grad;
  model.eval();

  double total_loss{0.0};
  int32_t num_correct{0};

  for (const auto& batch : data_loader) {
    const torch::Tensor data{batch.data.to(device)};
    const torch::Tensor label{batch.target.to(device)};

    const torch::Tensor out{model.forward(data)};

    const torch::Tensor loss{torch::nll_loss(out, label)};

    total_loss += loss.sum().item().to<double>();

    if (out.argmax(1).item().to<int32_t>() == label.item().to<int32_t>()) {
      ++num_correct;
    }
  }

  std::cout << "loss " << total_loss << " #correct " << num_correct
            << std::endl;
}

int main() {
  const torch::Device device{torch::cuda::is_available() ? torch::kCUDA
                                                         : torch::kCPU};

  const std::string path_data{"../data/MNIST/raw"};
  constexpr double mean{0.1307};
  constexpr double std{0.3081};
  const torch::data::transforms::Normalize<> transform(mean, std);
  auto train_data{torch::data::datasets::MNIST(path_data).map(transform).map(
      torch::data::transforms::Stack<>())};
  auto test_data{torch::data::datasets::MNIST(
                     path_data, torch::data::datasets::MNIST::Mode::kTest)
                     .map(transform)
                     .map(torch::data::transforms::Stack<>())};
  auto train_loader{torch::data::make_data_loader(train_data, 16)};
  auto test_loader{torch::data::make_data_loader(test_data, 1)};

  DNN model;
  model->to(device);

  torch::optim::SGD optimizer{model->parameters(),
                              torch::optim::SGDOptions(0.005).momentum(0.95)};

  test(*model, device, *test_loader);
  for (int32_t epoch = 0; epoch < 10; ++epoch) {
    train(*model, device, *train_loader, optimizer);
    test(*model, device, *test_loader);
  }

  torch::save(model, "mnist_model.pt");

  return EXIT_SUCCESS;
}