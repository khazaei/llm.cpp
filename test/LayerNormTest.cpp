//
// Created by Hamidreza Khazaei on 4/19/24.
//

#include <array>
#include <mdspan>

#include "LayerNorm.h"
#include "catch2/catch_test_macros.hpp"

constexpr auto batchDim = size_t{3};
constexpr auto sentenceLength = size_t{5};
constexpr auto embeddingDim = size_t{10};
constexpr auto totalLen = batchDim * sentenceLength * embeddingDim;

constexpr auto in = std::array<float, totalLen>{
    -1.1258398e+00, -1.1523602e+00, -2.5057858e-01, -4.3387881e-01, 8.4871036e-01,
    6.9200915e-01,  -3.1601277e-01, -2.1152194e+00, 3.2227492e-01,  -1.2633348e+00,
    3.4998319e-01,  3.0813393e-01,  1.1984151e-01,  1.2376579e+00,  1.1167772e+00,
    -2.4727815e-01, -1.3526537e+00, -1.6959312e+00, 5.6665063e-01,  7.9350835e-01,
    5.9883946e-01,  -1.5550951e+00, -3.4136039e-01, 1.8530061e+00,  7.5018948e-01,
    -5.8549756e-01, -1.7339675e-01, 1.8347794e-01,  1.3893661e+00,  1.5863342e+00,
    9.4629836e-01,  -8.4367675e-01, -6.1358309e-01, 3.1592742e-02,  -4.9267697e-01,
    2.4841475e-01,  4.3969584e-01,  1.1241119e-01,  6.4079237e-01,  4.4115627e-01,
    -1.0230965e-01, 7.9244399e-01,  -2.8966770e-01, 5.2507486e-02,  5.2286047e-01,
    2.3022053e+00,  -1.4688939e+00, -1.5866888e+00, -6.7308992e-01, 8.7283123e-01,

    1.0553575e+00,  1.7784372e-01,  -2.3033547e-01, -3.9175439e-01, 5.4329473e-01,
    -3.9515755e-01, -4.4621718e-01, 7.4402070e-01,  1.5209795e+00,  3.4105027e+00,
    -1.5311843e+00, -1.2341350e+00, 1.8197253e+00,  -5.5152869e-01, -5.6924808e-01,
    9.1997141e-01,  1.1108161e+00,  1.2898741e+00,  -1.4781740e+00, 2.5672328e+00,
    -4.7311980e-01, 3.3555076e-01,  -1.6293260e+00, -5.4974365e-01, -4.7983426e-01,
    -4.9968153e-01, -1.0669804e+00, 1.1149396e+00,  -1.4067143e-01, 8.0575359e-01,
    -9.3348235e-02, 6.8705022e-01,  -8.3831537e-01, 8.9182175e-04,  8.4189409e-01,
    -4.0003416e-01, 1.0394620e+00,  3.5815310e-01,  -2.4600095e-01, 2.3025165e+00,
    -1.8816892e+00, -4.9727023e-02, -1.0449786e+00, -9.5650083e-01, 3.3531860e-02,
    7.1008658e-01,  1.6458670e+00,  -1.3601689e+00, 3.4456542e-01,  5.1986772e-01,

    -2.6133225e+00, -1.6964744e+00, -2.2824179e-01, 2.7995500e-01,  2.4692640e-01,
    7.6887004e-02,  3.3800581e-01,  4.5440176e-01,  4.5694014e-01,  -8.6537135e-01,
    7.8130794e-01,  -9.2678940e-01, -2.1883368e-01, -2.4350653e+00, -7.2914757e-02,
    -3.3986408e-02, 9.6251827e-01,  3.4916824e-01,  -9.2146200e-01, -5.6194786e-02,
    -6.2269849e-01, -4.6372217e-01, 1.9217824e+00,  -4.0254554e-01, 1.2390248e-01,
    1.1647835e+00,  9.2337352e-01,  1.3872952e+00,  -8.8337576e-01, -4.1891348e-01,
    -8.0482650e-01, 5.6560963e-01,  6.1036462e-01,  4.6688354e-01,  -2.0106392e-01,
    -1.1792699e-01, -8.2936686e-01, -1.4072566e+00, 1.6268467e+00,  1.7227320e-01,
    -1.6115024e+00, -4.7944778e-01, 1.5739289e-01,  3.8536271e-01,  5.7365459e-01,
    9.9793130e-01,  5.4360944e-01,  7.8804389e-02,  8.6286008e-01,  -1.9489635e-02};

constexpr auto bias = std::array<float, embeddingDim>{
    0.4134675, 0.9219105, -1.3675891, -0.1525318, -0.9588874,
    0.7916716, 0.3360150, 1.0042887,  1.4007128,  -1.7721887};

constexpr auto weight = std::array<float, embeddingDim>{
    1.5151961, 0.3698294,  0.2814750,  0.5603455, 0.6310731,
    1.3468844, -0.4519350, -0.4052854, 1.4875096, -0.2229669};

const auto out = std::vector<float>{
    -6.8014061e-01, 6.4403117e-01,  -1.2956674e+00, -1.2403681e-01, -2.3047555e-02,
    2.5533540e+00,  2.5355646e-01,  1.7445250e+00,  2.7322421e+00,  -1.5770301e+00,
    7.9009211e-01,  9.9713355e-01,  -1.3675367e+00, 5.2357060e-01,  -2.7977607e-01,
    2.5827196e-01,  1.0541366e+00,  1.7984344e+00,  2.1182899e+00,  -1.9343382e+00,
    7.5309038e-01,  2.2255601e-01,  -1.5643772e+00, 6.6318333e-01,  -7.2364217e-01,
    -4.7288340e-01, 5.7743472e-01,  1.0787560e+00,  2.8888793e+00,  -2.0383809e+00,
    2.7724974e+00,  2.9262000e-01,  -1.7286386e+00, -2.1317394e-01, -1.6294701e+00,
    1.1775293e+00,  4.9175829e-02,  9.8852319e-01,  2.8893659e+00,  -1.9142965e+00,
    2.1496688e-01,  1.1734052e+00,  -1.4522665e+00, -1.4730658e-01, -6.8394828e-01,
    3.5508032e+00,  9.5504200e-01,  1.6026921e+00,  4.3623909e-01,  -1.9400593e+00,

    1.0221230e+00,  7.8490067e-01,  -1.5729659e+00, -6.4097697e-01, -9.8973984e-01,
    -3.8642141e-01, 7.5161898e-01,  9.5251751e-01,  2.6077154e+00,  -2.3238337e+00,
    -1.4887094e+00, 5.3574264e-01,  -1.0502778e+00, -4.6565321e-01, -1.3194827e+00,
    1.4483213e+00,  5.4353014e-02,  7.0009869e-01,  -4.1063648e-01, -2.1420557e+00,
    -1.3396892e-03, 1.2018170e+00,  -1.8594110e+00, -3.6065462e-01, -1.1370533e+00,
    3.7734774e-01,  8.0178636e-01,  2.9497737e-01,  1.6237311e+00,  -2.0745559e+00,
    -4.0036288e-01, 1.0613139e+00,  -1.7643745e+00, -3.9164937e-01, -6.0655755e-01,
    -4.1557071e-01, -2.0881221e-02, 1.0076466e+00,  3.3579168e-01,  -2.2781174e+00,
    -2.0571322e+00, 9.7732848e-01,  -1.5976638e+00, -5.6236976e-01, -8.1325924e-01,
    1.9880733e+00,  -4.7643453e-01, 1.4597110e+00,  2.1936166e+00,  -1.9290255e+00,

    -3.0215626e+00, 4.2388022e-01,  -1.3317631e+00, 2.0465907e-01,  -5.7753575e-01,
    1.3756697e+00,  2.1593118e-02,  6.7496550e-01,  2.6132107e+00,  -1.6579579e+00,
    2.1012664e+00,  6.5631217e-01,  -1.3559985e+00, -1.4614528e+00, -8.3413154e-01,
    1.1141729e+00,  -2.5524163e-01, 7.4068773e-01,  3.4093642e-01,  -1.8202653e+00,
    -1.0317658e+00, 6.3176805e-01,  -8.7337029e-01, -5.5563480e-01, -1.0590783e+00,
    2.0707834e+00,  2.3003832e-02,  5.2336204e-01,  -4.3104196e-01, -1.6079037e+00,
    -1.0634423e+00, 1.1690927e+00,  -1.1643564e+00, 1.5565789e-01,  -1.1171882e+00,
    5.8806872e-01,  7.8982764e-01,  1.6920670e+00,  4.2875977e+00,  -1.8160626e+00,
    -3.3034384e+00, 5.9808564e-01,  -1.3642648e+00, 3.2089889e-02,  -5.8538258e-01,
    2.3851328e+00,  8.7455310e-02,  1.0438852e+00,  2.8805690e+00,  -1.7198651e+00};

TEST_CASE("Test mean and variance.") {
  const auto m = llm::mean(in);
  CHECK(llm::isApproximatelyEqual(m, 0.0465393));
  const auto v = llm::variance(in, m);
  constexpr auto eps = 1e-6;
  CHECK(llm::isApproximatelyEqual(v, 1.0097520, eps));
}

TEST_CASE("Test layer norm.") {
  const auto inView = std::mdspan{in.data(), batchDim, sentenceLength, embeddingDim};
  auto outLayerNorm = std::vector<float>(totalLen);
  auto outView = std::mdspan{outLayerNorm.data(), batchDim, sentenceLength, embeddingDim};

  llm::layerNorm(outView, inView, weight, bias);
  constexpr auto eps = 1e-6;
  CHECK(llm::isTensorsEqual(out, outLayerNorm, eps));
}
