/**
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React from "react";
import { Icon, Paper, Typography } from "@mui/material";
import { Stack } from "@mui/system";

export default function({onClick, children}) {
  return (
    <Paper
      elevation={0}
      sx={{
        width: 'fit-content',
        minWidth: "40%",
        background: "#ffff",
        borderRadius: "50px",
        padding: "12px 12px",
        gap: "10px",
        // color: "#beff7",
        borderColor: "#AE929C",
        borderStyle: "solid",
        borderWidth: "1px",
        cursor: "pointer",

      }}
      onClick={onClick && onClick}
   >
      <Stack direction="row" spacing={0.5}>
        <Icon>
          <svg style={{marginBottom: "4px"}} xmlns="http://www.w3.org/2000/svg" width="10" height="14" viewBox="0 0 10 14" fill="none">
            <path d="M2.2077 9.53716C2.10917 9.40578 2.03732 9.26414 1.99216 9.11224C1.947 8.95624 1.92442 8.7756 1.92442 8.57032C1.92442 8.28705 1.85258 8.01404 1.70889 7.75129C1.5652 7.48854 1.39893 7.21142 1.21007 6.91994C1.02122 6.62434 0.854954 6.28975 0.711263 5.91616C0.567573 5.53845 0.495728 5.09507 0.495728 4.58599C0.495728 4.03997 0.606575 3.53295 0.828268 3.06492C1.05407 2.5928 1.36198 2.1802 1.75199 1.82713C2.14612 1.46996 2.60182 1.19284 3.11911 0.995781C3.63639 0.794614 4.19063 0.694031 4.78181 0.694031C5.37299 0.694031 5.92723 0.794614 6.44452 0.995781C6.9618 1.19284 7.41545 1.46996 7.80547 1.82713C8.19959 2.1802 8.5075 2.5928 8.7292 3.06492C8.95499 3.53295 9.06789 4.03997 9.06789 4.58599C9.06789 5.09507 8.99605 5.53845 8.85236 5.91616C8.70867 6.28975 8.5424 6.62434 8.35355 6.91994C8.1647 7.21142 7.99843 7.48854 7.85474 7.75129C7.71105 8.01404 7.6392 8.28705 7.6392 8.57032C7.6392 8.7756 7.61662 8.95624 7.57146 9.11224C7.5263 9.26414 7.45445 9.40578 7.35592 9.53716L6.47531 9.42631C6.63542 9.28262 6.74627 9.15946 6.80785 9.05682C6.86943 8.95008 6.90022 8.78791 6.90022 8.57032C6.90022 8.19262 6.97207 7.84982 7.11576 7.54191C7.25945 7.2299 7.42572 6.92609 7.61457 6.6305C7.80342 6.33081 7.96969 6.01879 8.11338 5.69446C8.25707 5.36603 8.32891 4.99654 8.32891 4.58599C8.32891 4.1426 8.23654 3.73001 8.0518 3.3482C7.86705 2.96639 7.61251 2.6318 7.28818 2.34442C6.96385 2.05704 6.58615 1.83329 6.15508 1.67318C5.72812 1.51307 5.27036 1.43301 4.78181 1.43301C4.29326 1.43301 3.83345 1.51307 3.40238 1.67318C2.97542 1.83329 2.59977 2.05704 2.27544 2.34442C1.95111 2.6318 1.69657 2.96639 1.51182 3.3482C1.32708 3.73001 1.23471 4.1426 1.23471 4.58599C1.23471 5.00064 1.30655 5.37218 1.45024 5.70062C1.59393 6.02905 1.7602 6.34107 1.94905 6.63666C2.13791 6.93225 2.30418 7.23606 2.44787 7.54807C2.59156 7.85598 2.6634 8.19673 2.6634 8.57032C2.6634 8.78791 2.69419 8.95008 2.75577 9.05682C2.81736 9.15946 2.9282 9.28262 3.08832 9.42631L2.2077 9.53716ZM4.78181 13.6262C4.35895 13.6262 4.03667 13.5235 3.81498 13.3183C3.59328 13.113 3.48244 12.8482 3.48244 12.5239C3.68771 12.6758 3.95456 12.7764 4.283 12.8256C4.61554 12.879 4.94603 12.879 5.27446 12.8256C5.60701 12.7764 5.87591 12.6758 6.08118 12.5239C6.08118 12.8482 5.97034 13.113 5.74864 13.3183C5.53105 13.5235 5.20878 13.6262 4.78181 13.6262ZM4.78181 12.647V12.1544L6.35214 11.9081V12.4007L4.78181 12.647ZM3.21764 11.9388V11.4462L6.88175 10.8673V11.36L3.21764 11.9388ZM4.78181 13.1397C4.33432 13.1397 3.92377 13.0719 3.55018 12.9365C3.18069 12.801 2.95284 12.606 2.86662 12.3514L2.54024 10.8427C2.40065 10.6744 2.30212 10.5101 2.24465 10.35C2.18717 10.1858 2.15843 9.9826 2.15843 9.74038H2.89741C2.89741 9.92923 2.91589 10.0668 2.95284 10.153C2.99389 10.2351 3.08216 10.348 3.21764 10.4917L3.57481 12.2467C3.61997 12.3001 3.76571 12.3596 4.01204 12.4253C4.25837 12.491 4.51496 12.5239 4.78181 12.5239C5.04866 12.5239 5.30525 12.491 5.55158 12.4253C5.79791 12.3596 5.94365 12.3001 5.98881 12.2467L6.34598 10.4917C6.48146 10.348 6.56768 10.2351 6.60463 10.153C6.64568 10.0668 6.66621 9.92923 6.66621 9.74038H7.40519C7.40519 9.9826 7.37645 10.1858 7.31898 10.35C7.2656 10.5101 7.16707 10.6744 7.02338 10.8427L6.697 12.3514C6.61079 12.606 6.38293 12.801 6.01344 12.9365C5.64395 13.0719 5.23341 13.1397 4.78181 13.1397ZM3.21764 10.9843V10.4917L6.88175 9.9128V10.4055L3.21764 10.9843ZM4.78181 10.2761C4.17421 10.2761 3.64255 10.2269 3.18685 10.1283C2.73525 10.0298 2.37602 9.89228 2.10917 9.71574C2.06401 9.6829 2.01885 9.63363 1.97369 9.56795C1.92853 9.50226 1.90595 9.41194 1.90595 9.29699C1.90595 9.21488 1.93263 9.13893 1.986 9.06914C2.03937 8.99934 2.11738 8.96445 2.22001 8.96445C2.2816 8.96445 2.32881 8.97882 2.36165 9.00755C2.58745 9.20462 2.90357 9.33804 3.31001 9.40784C3.71645 9.47763 4.20705 9.51252 4.78181 9.51252C5.35657 9.51252 5.84717 9.47763 6.25361 9.40784C6.66005 9.33804 6.97617 9.20462 7.20197 9.00755C7.23481 8.97882 7.28203 8.96445 7.34361 8.96445C7.45035 8.96445 7.52835 8.99934 7.57762 9.06914C7.63099 9.13893 7.65767 9.21488 7.65767 9.29699C7.65767 9.41194 7.63509 9.50226 7.58993 9.56795C7.54477 9.63363 7.49961 9.6829 7.45445 9.71574C7.1876 9.89228 6.82837 10.0298 6.37678 10.1283C5.92518 10.2269 5.39352 10.2761 4.78181 10.2761Z" fill="#614646"/>
          </svg>
        </Icon>
        <div style={{fontFamily:"Google Sans", fontSize:"14px", lineHeight: "18px"}}>
          {children}
        </div>
      </Stack>

    </Paper>
  )
}
