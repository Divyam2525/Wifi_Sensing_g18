#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"

#define WIFI_SSID      "POCO F5"   // 🔹 Replace with your Wi-Fi name
#define WIFI_PASSWORD  "1709_Divy_Thakk"   // 🔹 Replace with your Wi-Fi password
#define WIFI_CHANNEL   10                    // Match your hotspot/router channel

// Forward declarations
void promiscuous_callback(void *buf, wifi_promiscuous_pkt_type_t type);
void csi_callback(void *ctx, wifi_csi_info_t *data);

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n🔧 Initializing CSI capture with Wi-Fi connection...");

  // Connect to your Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ Wi-Fi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Set fixed Wi-Fi channel to match your AP (optional but recommended)
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

  // Enable promiscuous (sniffer) mode
  wifi_promiscuous_filter_t filter = {
    .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA
  };
  esp_wifi_set_promiscuous_filter(&filter);
  esp_wifi_set_promiscuous_rx_cb(&promiscuous_callback);
  esp_wifi_set_promiscuous(true);

  // Configure CSI capture
  wifi_csi_config_t csi_config = {
    .lltf_en = true,
    .htltf_en = true,
    .stbc_htltf2_en = true,
    .ltf_merge_en = true,
    .channel_filter_en = true,
    .manu_scale = false,
    .shift = false
  };

  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(&csi_callback, NULL);
  esp_wifi_set_csi(true);

  Serial.printf("✅ CSI capture started on channel %d (2.4 GHz)\n", WIFI_CHANNEL);
}

void loop() {
  delay(2000);
  Serial.println("Listening for CSI packets...");
}

// ----------------------------------------------------------------------
// Promiscuous mode callback (required, even if not used directly)
// ----------------------------------------------------------------------
void promiscuous_callback(void *buf, wifi_promiscuous_pkt_type_t type) {
  (void)buf;
  (void)type;
}

// ----------------------------------------------------------------------
// CSI callback — prints amplitude and phase information
// ----------------------------------------------------------------------
void csi_callback(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;

  Serial.print("📡 MAC: ");
  for (int i = 0; i < 6; i++) {
    Serial.printf("%02X", data->mac[i]);
    if (i < 5) Serial.print(":");
  }

  Serial.printf(" | RSSI: %d | CSI_len: %d | CSI: [", data->rx_ctrl.rssi, data->len);

  for (int i = 0; i < data->len; i += 2) {
    int8_t real = data->buf[i];
    int8_t imag = data->buf[i + 1];
    float amplitude = sqrt(real * real + imag * imag);
    float phase = atan2(imag, real);
    Serial.printf("%.2f,%.2f", amplitude, phase);
    if (i < data->len - 2) Serial.print("; ");
  }
  Serial.println("]");
  delay(2000);
}
