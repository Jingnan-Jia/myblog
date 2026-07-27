import { createClient } from "@supabase/supabase-js";

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.PUBLIC_SUPABASE_ANON_KEY!
);

const EMAILJS = {
  publicKey: process.env.PUBLIC_EMAILJS_PUBLIC_KEY!,
  serviceId: process.env.PUBLIC_EMAILJS_SERVICE_ID!,
  templateId: process.env.PUBLIC_EMAILJS_TEMPLATE_ID!,
};

export async function POST({ request }: { request: Request }) {
  try {
    const body = await request.json();
    const nickname = String(body.nickname || "匿名").slice(0, 50);
    const content = String(body.content || "").slice(0, 100000);

    if (!content) {
      return new Response(JSON.stringify({ error: "content is required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // 写入 Supabase
    const { error: dbError } = await supabase.from("messages").insert({
      nickname,
      content,
      parent_id: null,
      is_admin: false,
    });

    if (dbError) throw dbError;

    // 发邮件通知
    await fetch("https://api.emailjs.com/api/v1.0/email/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        service_id: EMAILJS.serviceId,
        template_id: EMAILJS.templateId,
        user_id: EMAILJS.publicKey,
        template_params: {
          name: nickname,
          title: `新留言: ${content.slice(0, 100)}`,
          email: "guestbook@jiajingnan.cn",
          message: content,
        },
      }),
    });

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: any) {
    return new Response(
      JSON.stringify({ error: err.message || "internal error" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
