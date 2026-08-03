#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/manuals}"
mkdir -p "$ROOT"

download() {
  local category="$1" filename="$2" url="$3"
  mkdir -p "$ROOT/$category"
  echo "Downloading $category/$filename"
  curl -fL -A "Mozilla/5.0" --retry 3 --retry-delay 2 "$url" -o "$ROOT/$category/$filename"
}

download "computers" "lenovo-thinkpad-t480-hardware-maintenance-manual.pdf" "https://download.lenovo.com/pccbbs/mobiles_pdf/t480_hmm_en.pdf"
download "computers" "lenovo-thinkpad-t480s-hardware-maintenance-manual.pdf" "https://download.lenovo.com/pccbbs/mobiles_pdf/t480s-hmm_en.pdf"
download "computers" "lenovo-thinkpad-t14-gen-2-p14s-gen-2-hardware-maintenance-manual.pdf" "https://download.lenovo.com/pccbbs/mobiles_pdf/t14_gen2_p14s_gen2_hmm_en.pdf"
download "computers" "lenovo-thinkpad-t14-gen-3-p14s-gen-3-hardware-maintenance-manual.pdf" "https://download.lenovo.com/pccbbs/mobiles_pdf/t14_gen3_p14s_gen3_hmm_en.pdf"
download "computers" "dell-latitude-7490-owner-s-manual.pdf" "https://dl.dell.com/topicspdf/latitude-14-7490-laptop_owners-manual2_en-us.pdf"
download "computers" "dell-optiplex-7060-sff-service-manual.pdf" "https://dl.dell.com/topicspdf/optiplex-7060-desktop_service-manual2_en-us.pdf"
download "computers" "hp-elitebook-855-g8-maintenance-and-service-guide.pdf" "https://h10032.www1.hp.com/ctg/Manual/c07067672.pdf"
download "computers" "hp-elitedesk-800-g5-sff-maintenance-and-service-guide.pdf" "https://h10032.www1.hp.com/ctg/Manual/c06443940.pdf"
download "routers" "tp-link-archer-c6-user-guide.pdf" "https://static.tp-link.com/upload/manual/2022/202206/20220607/1910013208_Archer%20C6%28US%29_UG_V1.pdf"
download "routers" "tp-link-archer-c1200-user-guide.pdf" "https://static.tp-link.com/res/down/doc/Archer_C1200%28US%29_V1_UG.pdf"
download "routers" "netgear-nighthawk-rax40-user-manual.pdf" "https://www.downloads.netgear.com/files/GDC/RAX40/RAX40_UM_EN.pdf"
download "routers" "netgear-orbi-rbk752-user-manual.pdf" "https://www.downloads.netgear.com/files/GDC/RBK752/RBK752_UM_EN.pdf"
download "routers" "asus-rt-ax3000-user-manual.pdf" "https://dlcdnets.asus.com/pub/ASUS/wireless/RT-AX3000/E16134_RT-AX3000_UM_1119.pdf"
download "routers" "asus-rt-be58u-user-manual.pdf" "https://dlcdnets.asus.com/pub/ASUS/wireless/RT-BE58U/E23503_RT-BE58U_UM_WEB.pdf"
download "routers" "linksys-mr7300-series-user-guide.pdf" "https://downloads.linksys.com/support/assets/userguide/USER%20GUIDE%20-%20MR7300%20Series%20-%20INTL_B00.pdf"
download "routers" "linksys-e9450-user-guide.pdf" "https://downloads.linksys.com/support/assets/userguide/E9450_USERGUIDE_EN%20LNKPG-00832%20Rev%20A00.pdf"
download "printers" "brother-hl-l2350dw-series-online-user-s-guide.pdf" "https://download.brother.com/welcome/doc100810/cv_hll2310d_uke_oug_c.pdf"
download "printers" "brother-hl-l2350dw-series-product-safety-guide.pdf" "https://download.brother.com/welcome/doc100759/cv_hll2310d_use_psg_e.pdf"
download "printers" "epson-et-2800-et-2803-user-s-guide.pdf" "https://files.support.epson.com/docid/cpd6/cpd60271.pdf"
download "printers" "epson-l3150-user-s-guide.pdf" "https://files.support.epson.com/docid/cpd5/cpd55466.pdf"
download "printers" "canon-g2000-series-online-manual.pdf" "https://downloads.canon.com/hsg2023/ijprinters/manuals/G2000ser_OnlineManual_Win_EN_V02.pdf"
download "printers" "canon-ts5300-series-online-manual.pdf" "https://downloads.canon.com/hsg2023/ijprinters/manuals/TS5300ser_OnlineManual_Mac_EN_V01.pdf"
download "printers" "hp-laserjet-pro-m404-m405-user-guide.pdf" "https://h10032.www1.hp.com/ctg/Manual/c06177490.pdf"
download "printers" "hp-laserjet-1022-series-service-manual.pdf" "https://h10032.www1.hp.com/ctg/Manual/c00631427.pdf"

echo "Downloaded manuals to $ROOT"
find "$ROOT" -type f -name "*.pdf" | sort
